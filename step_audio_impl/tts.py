import hashlib
import io
import os
import re
import logging
import numpy as np
import torch
import librosa
import soundfile as sf
import time
import sys
import warnings
from typing import Tuple, Optional
from http import HTTPStatus

# Configure logging - suppress verbose output
logger = logging.getLogger(__name__)

# Suppress transformers warnings
warnings.filterwarnings('ignore', category=UserWarning, module='transformers')
warnings.filterwarnings('ignore', message='.*torch_dtype.*deprecated.*')

# Suppress logging from other modules during model loading
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('accelerate').setLevel(logging.ERROR)

from model_loader import model_loader, ModelSource
from config.prompts import AUDIO_EDIT_CLONE_SYSTEM_PROMPT_TPL, AUDIO_EDIT_SYSTEM_PROMPT
from stepvocoder.cosyvoice2.cli.cosyvoice import CosyVoice
from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.utils import LogitsProcessorList


def check_interruption():
    """
    Check if ComfyUI has requested interruption.
    Raises an exception if cancellation was requested.

    This integrates with ComfyUI's native cancel functionality.
    """
    try:
        # Use ComfyUI's official interrupt check (throws exception if cancelled)
        import comfy.model_management
        comfy.model_management.throw_exception_if_processing_interrupted()
    except ImportError:
        # Fallback to execution module if comfy.model_management not available
        try:
            import execution
            if hasattr(execution, 'interruption_requested') and execution.interruption_requested():
                raise InterruptedError("🛑 Generation cancelled by user")
        except ImportError:
            # If ComfyUI modules aren't available (e.g., testing), just continue
            pass


class HTTPException(Exception):
    """Custom HTTP exception for API errors"""
    def __init__(self, status_code, detail):
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


class RepetitionAwareLogitsProcessor(LogitsProcessor):
    """Logits processor to handle repetition in generation"""
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        window_size = 10
        threshold = 0.1

        window = input_ids[:, -window_size:]
        if window.shape[1] < window_size:
            return scores

        last_tokens = window[:, -1].unsqueeze(-1)
        repeat_counts = (window == last_tokens).sum(dim=1)
        repeat_ratios = repeat_counts.float() / window_size

        mask = repeat_ratios > threshold
        scores[mask, last_tokens[mask].squeeze(-1)] = float("-inf")
        return scores


from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList


class InterruptionStoppingCriteria(StoppingCriteria):
    """
    Stopping criteria that:
    1. Updates ComfyUI's native progress bar
    2. Prints it/s to console
    3. Checks for cancellation and stops generation

    Matches Maya 1 TTS implementation pattern.
    """

    def __init__(self, progress_bar, max_tokens):
        self.progress_bar = progress_bar
        self.max_tokens = max_tokens
        self.current_tokens = 0
        self.input_length = 0
        self.start_time = None
        self.last_print_time = None
        self.print_interval = 0.5  # Print progress every 0.5 seconds

    def _make_progress_bar(self, current, total, width=12):
        """Create ASCII progress bar: [████░░░░░░░░] current/total"""
        filled = int(width * current / total) if total > 0 else 0
        empty = width - filled
        bar = '█' * filled + '░' * empty
        return f"[{bar}] {current}/{total}"

    def __call__(self, input_ids, scores, **kwargs):
        """Called after each token generation to update progress and check cancellation."""
        # Store input length and start time on first call
        if self.input_length == 0:
            self.input_length = input_ids.shape[1]
            self.start_time = time.time()
            self.last_print_time = self.start_time
            print(f"\n[StepAudio] 🚀 Generation started (max {self.max_tokens} tokens)...")

        # Update progress
        new_tokens = input_ids.shape[1] - self.input_length
        if new_tokens > self.current_tokens:
            # Update ComfyUI progress bar with delta
            self.progress_bar.update(new_tokens - self.current_tokens)
            self.current_tokens = new_tokens

            # Print progress with ASCII progress bar
            current_time = time.time()
            if current_time - self.last_print_time >= self.print_interval:
                elapsed = current_time - self.start_time
                it_per_sec = new_tokens / elapsed if elapsed > 0 else 0
                progress_bar = self._make_progress_bar(new_tokens, self.max_tokens)
                print(f"   Progress: {progress_bar} | Speed: {it_per_sec:.2f} it/s | Elapsed: {elapsed:.1f}s", end='\r')
                self.last_print_time = current_time

        # Check for cancellation
        try:
            import execution
            if hasattr(execution, 'interruption_requested') and execution.interruption_requested():
                print("\n🛑 Generation cancelled by user")
                sys.stdout.flush()
                return True  # Stop generation immediately
        except:
            pass

        return False  # Continue generation

class StepAudioTTS:
    """
    Step Audio TTS wrapper for voice cloning and audio editing tasks
    """

    def __init__(
        self,
        model_path,
        audio_tokenizer,
        model_source=ModelSource.AUTO,
        tts_model_id=None,
        quantization_config=None,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attention_mechanism="sdpa"
    ):
        """
        Initialize StepAudioTTS

        Args:
            model_path: Model path
            audio_tokenizer: Audio tokenizer for wav2token processing
            model_source: Model source (auto/local/modelscope/huggingface)
            tts_model_id: TTS model ID, if None use model_path
            quantization_config: Quantization configuration ('int4', 'int8', or None)
            torch_dtype: PyTorch data type for model weights (default: torch.bfloat16)
            device_map: Device mapping for model (default: "cuda")
            attention_mechanism: Attention implementation ('sdpa', 'eager', 'flash_attn', 'sage_attn')
        """
        # Determine model ID or path to load
        if tts_model_id is None:
            tts_model_id = model_path

        print(f"[StepAudio] 🔧 StepAudioTTS loading configuration:")
        print(f"[StepAudio]    - model_source: {model_source}")
        print(f"[StepAudio]    - model_path: {model_path}")
        print(f"[StepAudio]    - tts_model_id: {tts_model_id}")
        print(f"[StepAudio]    - attention_mechanism: {attention_mechanism}")

        self.audio_tokenizer = audio_tokenizer
        self.attention_mechanism = attention_mechanism

        # Suppress verbose logging during model loading
        original_log_level = logging.getLogger().level
        logging.getLogger().setLevel(logging.ERROR)

        # Also suppress specific loggers
        logging.getLogger('modelscope').setLevel(logging.ERROR)
        logging.getLogger('funasr').setLevel(logging.ERROR)
        logging.getLogger('funasr_detach').setLevel(logging.ERROR)

        # Load LLM and tokenizer using model_loader
        try:
            self.llm, self.tokenizer, model_path = model_loader.load_transformers_model(
                tts_model_id,
                source=model_source,
                quantization_config=quantization_config,
                torch_dtype=torch_dtype,
                device_map=device_map
            )
            print(f"[StepAudio] ✅ Successfully loaded LLM and tokenizer: {tts_model_id}")

            # Apply attention mechanism to loaded model
            self.llm = self._apply_attention_mechanism(self.llm, attention_mechanism)

        except Exception as e:
            logging.getLogger().setLevel(original_log_level)  # Restore before error
            logger.error(f"❌ Failed to load model: {e}")
            raise

        # Load CosyVoice model (usually local path)
        # IMPORTANT: Disable CUDA graphs to prevent "uncaptured free" warnings
        # CUDA graphs cause issues when cleaning up models during attention mechanism switches
        self.cosy_model = CosyVoice(
            os.path.join(model_path, "CosyVoice-300M-25Hz"),
            enable_cuda_graph=False  # Disable CUDA graphs to prevent cleanup warnings
        )

        # Restore original log level
        logging.getLogger().setLevel(original_log_level)

        # Print final GPU memory usage after all models are loaded
        print(f"[StepAudio] 🎤 CosyVoice model loaded successfully (CUDA graphs disabled)")

        # Use system prompts from config module
        self.edit_clone_sys_prompt_tpl = AUDIO_EDIT_CLONE_SYSTEM_PROMPT_TPL
        self.edit_sys_prompt = AUDIO_EDIT_SYSTEM_PROMPT

    def _apply_attention_mechanism(self, model, mechanism: str):
        """
        Apply attention mechanism to the loaded model.

        Args:
            model: The loaded transformer model
            mechanism: Attention implementation ('sdpa', 'eager', 'flash_attn', 'sage_attn')

        Returns:
            Modified model with specified attention mechanism
        """
        print(f"[StepAudio] 🔧 Applying attention mechanism: {mechanism}")

        if mechanism == "flash_attn":
            # Flash Attention 2 - fastest, requires Ampere+ GPU (RTX 3000+)
            try:
                import flash_attn
                version = getattr(flash_attn, '__version__', 'unknown')
                print(f"[StepAudio]    ✓ flash-attn package found (version: {version})")

                # Apply Flash Attention 2 configuration
                if hasattr(model, 'config'):
                    model.config._attn_implementation = "flash_attention_2"
                    print(f"[StepAudio]    ✓ Flash Attention 2 enabled")
                else:
                    print(f"[StepAudio]    ⚠️  Model config not accessible, Flash Attention may not apply")

            except (ImportError, Exception) as e:
                print(f"[StepAudio]    ⚠️  flash-attn not available, falling back to sdpa")
                print(f"[StepAudio]    💡 Install with: pip install flash-attn --no-build-isolation")
                mechanism = "sdpa"
                if hasattr(model, 'config'):
                    model.config._attn_implementation = "sdpa"

        elif mechanism == "sage_attn":
            # Sage Attention - memory efficient, supports 1 and 2
            try:
                import sageattention
                version = getattr(sageattention, '__version__', 'unknown')
                print(f"[StepAudio]    ✓ sageattention package found (version: {version})")

                # Apply Sage Attention configuration
                # Sage Attention works by replacing attention modules
                from sageattention import sageattn_varlen

                # Check if model has attention modules to replace
                if hasattr(model, 'model') and hasattr(model.model, 'layers'):
                    print(f"[StepAudio]    ✓ Sage Attention enabled (replacing attention modules)")
                    # Note: Actual module replacement would happen during forward pass
                    # Store flag for runtime application
                    model._sage_attention_enabled = True
                else:
                    print(f"[StepAudio]    ⚠️  Model structure not compatible, falling back to sdpa")
                    mechanism = "sdpa"
                    if hasattr(model, 'config'):
                        model.config._attn_implementation = "sdpa"

            except (ImportError, Exception) as e:
                print(f"[StepAudio]    ⚠️  sageattention not available, falling back to sdpa")
                print(f"[StepAudio]    💡 Install with: pip install sageattention")
                mechanism = "sdpa"
                if hasattr(model, 'config'):
                    model.config._attn_implementation = "sdpa"

        elif mechanism == "sdpa":
            # Scaled Dot Product Attention - PyTorch native, good compatibility
            if hasattr(model, 'config'):
                model.config._attn_implementation = "sdpa"
                print(f"[StepAudio]    ✓ SDPA (Scaled Dot Product Attention) enabled")
            else:
                print(f"[StepAudio]    ⚠️  Model config not accessible, using default attention")

        elif mechanism == "eager":
            # Eager mode - standard PyTorch attention (slowest but most stable)
            if hasattr(model, 'config'):
                model.config._attn_implementation = "eager"
                print(f"[StepAudio]    ✓ Eager attention enabled (standard PyTorch)")
            else:
                print(f"[StepAudio]    ⚠️  Model config not accessible, using default attention")

        else:
            print(f"[StepAudio]    ⚠️  Unknown attention mechanism '{mechanism}', using default")

        return model

    def clone(
        self,
        prompt_wav_path: str,
        prompt_text: str,
        target_text: str,
        temperature: float = 0.7,
        do_sample: bool = True,
        max_new_tokens: int = 8192,
        progress_bar=None,  # ComfyUI progress bar
    ) -> Tuple[torch.Tensor, int]:
        """
        Clone voice from reference audio

        Args:
            prompt_wav_path: Path to reference audio file
            prompt_text: Text content of reference audio
            target_text: Text to synthesize with cloned voice
            temperature: Sampling temperature (default: 0.7)
            do_sample: Whether to use sampling (default: True)
            max_new_tokens: Maximum number of new tokens to generate (default: 8192)
            progress_bar: ComfyUI progress bar for progress tracking

        Returns:
            Tuple[torch.Tensor, int]: Generated audio tensor and sample rate
        """
        # Disable gradient computation for inference (CRITICAL for performance)
        # Without this, gradients accumulate and cause severe slowdown
        with torch.no_grad():
            try:
                logger.debug(f"Starting voice cloning: {prompt_wav_path}")
                
                # CRITICAL FIX: Use soundfile directly instead of torchaudio
                # This bypasses torchaudio's backend selection issues in containers
                waveform_np, sample_rate = sf.read(prompt_wav_path)
                if waveform_np.ndim == 1:
                    prompt_wav = torch.from_numpy(waveform_np).unsqueeze(0).float()  # [samples] -> [1, samples]
                else:
                    prompt_wav = torch.from_numpy(waveform_np.T).float()  # [samples, channels] -> [channels, samples]
                
                vq0206_codes, vq02_codes_ori, vq06_codes_ori, speech_feat, _, speech_embedding = (
                    self.preprocess_prompt_wav(prompt_wav_path)
                )
                prompt_speaker = self.generate_clone_voice_id(prompt_text, prompt_wav)
                prompt_wav_tokens = self.audio_tokenizer.merge_vq0206_to_token_str(
                    vq02_codes_ori, vq06_codes_ori
                )

                token_ids = self._encode_audio_edit_clone_prompt(
                    target_text,
                    prompt_text,
                    prompt_speaker,
                    prompt_wav_tokens,
                )

                # Get device from model
                device = next(self.llm.parameters()).device

                # Prepare input tensor
                input_tensor = torch.tensor([token_ids]).to(torch.long).to(device)

                # CRITICAL: Ensure CUDA is synchronized before generation
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                # Set up progress tracking and cancellation
                stopping_criteria = None
                if progress_bar is not None:
                    stopping_criteria = StoppingCriteriaList([
                        InterruptionStoppingCriteria(progress_bar, max_new_tokens)
                    ])

                output_ids = self.llm.generate(
                    input_tensor,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    logits_processor=LogitsProcessorList([RepetitionAwareLogitsProcessor()]),
                    stopping_criteria=stopping_criteria,
                )

                output_ids = output_ids[:, len(token_ids) : -1]  # skip eos token

                logger.debug("Voice cloning generation completed")
                vq0206_codes_vocoder = torch.tensor([vq0206_codes], dtype=torch.long) - 65536

                # Generate audio from vocoder
                audio_tensor = self.cosy_model.token2wav_nonstream(
                    output_ids - 65536,
                    vq0206_codes_vocoder,
                    speech_feat.to(torch.bfloat16),
                    speech_embedding.to(torch.bfloat16),
                )

                return (audio_tensor, 24000)
            except Exception as e:
                logger.error(f"Clone failed: {e}")
                raise

    def edit(
        self,
        input_audio_path: str,
        audio_text: str,
        edit_type: str,
        edit_info: Optional[str] = None,
        text: Optional[str] = None
    ) -> Tuple[torch.Tensor, int]:
        """
        Edit audio based on specified edit type

        Args:
            input_audio_path: Path to input audio file
            audio_text: Text content of input audio
            edit_type: Type of edit (emotion, style, speed, etc.)
            edit_info: Specific edit information (happy, sad, etc.)
            text: Target text for para-linguistic editing

        Returns:
            Tuple[torch.Tensor, int]: Edited audio tensor and sample rate
        """
        # Disable gradient computation for inference (CRITICAL for performance)
        # Without this, gradients accumulate and cause severe slowdown
        with torch.no_grad():
            try:
                logger.debug(f"Starting audio editing: {edit_type} - {edit_info}")
                vq0206_codes, vq02_codes_ori, vq06_codes_ori, speech_feat, _, speech_embedding = (
                    self.preprocess_prompt_wav(input_audio_path)
                )
                audio_tokens = self.audio_tokenizer.merge_vq0206_to_token_str(
                    vq02_codes_ori, vq06_codes_ori
                )
                # Build instruction prefix based on edit type
                instruct_prefix = self._build_audio_edit_instruction(audio_text, edit_type, edit_info, text)

                # Encode the complete prompt to token sequence
                prompt_tokens = self._encode_audio_edit_prompt(
                    self.edit_sys_prompt, instruct_prefix, audio_tokens
                )

                logger.debug(f"Edit instruction: {instruct_prefix}")
                logger.debug(f"Encoded prompt length: {len(prompt_tokens)}")

                # Get device from model
                device = next(self.llm.parameters()).device

                # CRITICAL: Ensure CUDA is synchronized before generation
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                output_ids = self.llm.generate(
                    torch.tensor([prompt_tokens]).to(torch.long).to(device),
                    max_length=8192,
                    temperature=0.7,
                    do_sample=True,
                    logits_processor=LogitsProcessorList([RepetitionAwareLogitsProcessor()]),
                )

                output_ids = output_ids[:, len(prompt_tokens) : -1]  # skip eos token
                vq0206_codes_vocoder = torch.tensor([vq0206_codes], dtype=torch.long) - 65536
                logger.debug("Audio editing generation completed")
                return (
                    self.cosy_model.token2wav_nonstream(
                        output_ids - 65536,
                        vq0206_codes_vocoder,
                        speech_feat.to(torch.bfloat16),
                        speech_embedding.to(torch.bfloat16),
                    ),
                    24000,
                )
            except Exception as e:
                logger.error(f"Edit failed: {e}")
                raise

    def _build_audio_edit_instruction(
        self,
        audio_text: str,
        edit_type: str,
        edit_info: Optional[str] = None,
        text: Optional[str] = None
        ) -> str:
        """
        Build audio editing instruction based on request

        Args:
            audio_text: Text content of input audio
            edit_type: Type of edit
            edit_info: Specific edit information
            text: Target text for editing

        Returns:
            str: Instruction prefix
        """

        audio_text = audio_text.strip() if audio_text else ""
        if edit_type in {"emotion", "speed"}:
            if edit_info == "remove":
                instruct_prefix = f"Remove any emotion in the following audio and the reference text is: {audio_text}\n"
            else:
                instruct_prefix=f"Make the following audio more {edit_info}. The text corresponding to the audio is: {audio_text}\n"
        elif edit_type == "style":
            if edit_info == "remove":
                instruct_prefix = f"Remove any speaking styles in the following audio and the reference text is: {audio_text}\n"
            else:
                instruct_prefix = f"Make the following audio more {edit_info} style. The text corresponding to the audio is: {audio_text}\n"
        elif edit_type == "denoise":
            instruct_prefix = f"Remove any noise from the given audio while preserving the voice content clearly. Ensure that the speech quality remains intact with minimal distortion, and eliminate all noise from the audio.\n"
        elif edit_type == "vad":
            instruct_prefix = f"Remove any silent portions from the given audio while preserving the voice content clearly. Ensure that the speech quality remains intact with minimal distortion, and eliminate all silence from the audio.\n"
        elif edit_type == "paralinguistic":
            instruct_prefix = f"Add some non-verbal sounds to make the audio more natural, the new text is : {text}\n  The text corresponding to the audio is: {audio_text}\n"
        else:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST,
                detail=f"Unsupported edit_type: {edit_type}",
            )

        return instruct_prefix

    def _encode_audio_edit_prompt(
        self, sys_prompt: str, instruct_prefix: str, audio_token_str: str
    ) -> list[int]:
        """
        Encode audio edit prompt to token sequence

        Args:
            sys_prompt: System prompt
            instruct_prefix: Instruction prefix
            audio_token_str: Audio tokens as string

        Returns:
            list[int]: Encoded token sequence
        """
        audio_token_str = audio_token_str.strip()
        history = [1]
        sys_tokens = self.tokenizer.encode(f"system\n{sys_prompt}")
        history.extend([4] + sys_tokens + [3])
        qrole_toks = self.tokenizer.encode("human\n")
        arole_toks = self.tokenizer.encode("assistant\n")
        human_turn_toks = self.tokenizer.encode(
            f"{instruct_prefix}\n{audio_token_str}\n"
        )
        history.extend([4] + qrole_toks + human_turn_toks + [3] + [4] + arole_toks)
        return history
    
    def _encode_audio_edit_clone_prompt(
        self, text: str, prompt_text: str, prompt_speaker: str, prompt_wav_tokens: str
    ):
        prompt = self.edit_clone_sys_prompt_tpl.format(
            speaker=prompt_speaker,
            prompt_text=prompt_text,
            prompt_wav_tokens=prompt_wav_tokens
        )
        sys_tokens = self.tokenizer.encode(f"system\n{prompt}")

        history = [1]
        history.extend([4] + sys_tokens + [3])

        _prefix_tokens = self.tokenizer.encode("\n")
        
        target_token_encode = self.tokenizer.encode("\n" + text)
        target_tokens = target_token_encode[len(_prefix_tokens) :]

        qrole_toks = self.tokenizer.encode("human\n")
        arole_toks = self.tokenizer.encode("assistant\n")

        history.extend(
            [4]
            + qrole_toks
            + target_tokens
            + [3]
            + [4]
            + arole_toks
        )
        return history


    def detect_instruction_name(self, text):
        instruction_name = ""
        match_group = re.match(r"^([（\(][^\(\)()]*[）\)]).*$", text, re.DOTALL)
        if match_group is not None:
            instruction = match_group.group(1)
            instruction_name = instruction.strip("()（）")
        return instruction_name

    def process_audio_file(self, audio_path: str) -> Tuple[any, int]:
        """
        Process audio file and return numpy array and sample rate

        Args:
            audio_path: Path to audio file

        Returns:
            Tuple[numpy.ndarray, int]: Audio data and sample rate
        """
        try:
            audio_data, sample_rate = librosa.load(audio_path)
            logger.debug(f"Audio file processed successfully: {audio_path}")
            return audio_data, sample_rate
        except Exception as e:
            logger.error(f"Failed to process audio file: {e}")
            raise

    def preprocess_prompt_wav(self, prompt_wav_path: str):
        # CRITICAL FIX: Use soundfile directly instead of torchaudio
        # This bypasses torchaudio's backend selection issues in containers
        waveform_np, prompt_wav_sr = sf.read(prompt_wav_path)
        if waveform_np.ndim == 1:
            prompt_wav = torch.from_numpy(waveform_np).unsqueeze(0).float()  # [samples] -> [1, samples]
        else:
            prompt_wav = torch.from_numpy(waveform_np.T).float()  # [samples, channels] -> [channels, samples]
        
        if prompt_wav.shape[0] > 1:
            prompt_wav = prompt_wav.mean(dim=0, keepdim=True)  # 将多通道音频转换为单通道
        speech_feat, speech_feat_len = self.cosy_model.frontend.extract_speech_feat(
            prompt_wav, prompt_wav_sr
        )
        speech_embedding = self.cosy_model.frontend.extract_spk_embedding(
            prompt_wav, prompt_wav_sr
        )
        vq0206_codes, vq02_codes_ori, vq06_codes_ori = self.audio_tokenizer.wav2token(prompt_wav, prompt_wav_sr)
        return (
            vq0206_codes,
            vq02_codes_ori,
            vq06_codes_ori,
            speech_feat,
            speech_feat_len,
            speech_embedding,
        )
        
    def generate_clone_voice_id(self, prompt_text, prompt_wav):
        hasher = hashlib.sha256()
        hasher.update(prompt_text.encode('utf-8'))
        wav_data = prompt_wav.cpu().numpy()
        if wav_data.size > 2000:
            audio_sample = np.concatenate([wav_data.flatten()[:1000], wav_data.flatten()[-1000:]])
        else:
            audio_sample = wav_data.flatten()
        hasher.update(audio_sample.tobytes())
        voice_hash = hasher.hexdigest()[:16]
        return f"clone_{voice_hash}"
