"""
Model Manager for Step Audio EditX TTS
Handles model loading, caching, and VRAM management
"""

import os
import sys
import torch
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from pathlib import Path

# Add bundled Step Audio implementation to path
STEP_AUDIO_IMPL_DIR = Path(__file__).parent.parent / "step_audio_impl"
if str(STEP_AUDIO_IMPL_DIR) not in sys.path:
    sys.path.insert(0, str(STEP_AUDIO_IMPL_DIR))

try:
    from tts import StepAudioTTS
    from tokenizer import StepAudioTokenizer
    from model_loader import ModelSource, model_loader
except ImportError as e:
    print(f"[StepAudio] ERROR: Failed to import bundled Step Audio modules: {e}")
    print(f"[StepAudio] Make sure 'step_audio_impl' directory exists at: {STEP_AUDIO_IMPL_DIR}")
    StepAudioTTS = None
    StepAudioTokenizer = None
    ModelSource = None
    model_loader = None


@dataclass
class StepAudioModelConfig:
    """Configuration for Step Audio model loading."""
    model_path: str
    model_source: str = "local"  # 'local', 'modelscope', 'huggingface', 'auto'
    quantization: Optional[str] = None  # None, 'int4', 'int8', 'int4_awq'
    torch_dtype: str = "bfloat16"  # 'bfloat16', 'float16', 'float32'
    device_map: str = "auto"  # 'auto', 'cuda', 'cpu', or device mapping
    attention_mechanism: str = "sdpa"  # 'sdpa', 'eager', 'flash_attn', 'sage_attn'
    tts_model_id: Optional[str] = None  # Optional online model ID


class StepAudioModelManager:
    """
    Manages Step Audio model loading and caching.
    Implements singleton pattern with cache key validation.

    FIXED:
    - Auto-clears cache when config changes (dtype, quantization)
    - Flash Attention support (optional)
    - Uses bundled implementation (standalone)
    """

    _model_cache: Dict[str, "StepAudioModelWrapper"] = {}
    _tokenizer_cache: Dict[str, Any] = {}
    _last_config_key: Optional[str] = None  # Track config changes

    @staticmethod
    def get_cache_key(config: StepAudioModelConfig) -> str:
        """
        Generate cache key from model configuration.

        Args:
            config: Model configuration

        Returns:
            Cache key string
        """
        return f"{config.model_path}|{config.quantization}|{config.torch_dtype}|{config.device_map}|{config.attention_mechanism}"

    @staticmethod
    def _parse_torch_dtype(dtype_str: str) -> torch.dtype:
        """
        Convert dtype string to torch.dtype.

        Args:
            dtype_str: One of 'bfloat16', 'float16', 'float32'

        Returns:
            torch.dtype
        """
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
            "auto": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        }
        return dtype_map.get(dtype_str, torch.bfloat16)

    @staticmethod
    def _parse_model_source(source_str: str):
        """
        Convert source string to ModelSource enum.

        Args:
            source_str: One of 'auto', 'local', 'modelscope', 'huggingface'

        Returns:
            ModelSource enum value
        """
        if ModelSource is None:
            raise ImportError("ModelSource not available. Step Audio not properly imported.")

        source_map = {
            "auto": ModelSource.AUTO,
            "local": ModelSource.LOCAL,
            "modelscope": ModelSource.MODELSCOPE,
            "huggingface": ModelSource.HUGGINGFACE,
        }
        return source_map.get(source_str.lower(), ModelSource.AUTO)

    @classmethod
    def load_tokenizer(
        cls,
        encoder_path: str,
        model_source: str = "local",
        funasr_model_id: Optional[str] = None
    ) -> Any:
        """
        Load or retrieve cached Step Audio tokenizer.

        Args:
            encoder_path: Path to Step-Audio-Tokenizer
            model_source: Model source ('auto', 'local', 'modelscope', 'huggingface')
            funasr_model_id: Optional FunASR model ID

        Returns:
            StepAudioTokenizer instance
        """
        if StepAudioTokenizer is None:
            raise ImportError("StepAudioTokenizer not available. Step Audio not properly imported.")

        # Generate cache key
        cache_key = f"{encoder_path}|{model_source}|{funasr_model_id}"

        # Check cache
        if cache_key in cls._tokenizer_cache:
            print(f"[StepAudio] Using cached tokenizer")
            return cls._tokenizer_cache[cache_key]

        # Load tokenizer
        print(f"[StepAudio] Loading tokenizer from {encoder_path}")
        try:
            source = cls._parse_model_source(model_source)
            # Only pass funasr_model_id if provided (allows tokenizer to use its default)
            tokenizer_kwargs = {
                "encoder_path": encoder_path,
                "model_source": source,
            }
            if funasr_model_id is not None:
                tokenizer_kwargs["funasr_model_id"] = funasr_model_id

            tokenizer = StepAudioTokenizer(**tokenizer_kwargs)

            # Cache tokenizer
            cls._tokenizer_cache[cache_key] = tokenizer
            print(f"[StepAudio] Tokenizer loaded and cached")

            return tokenizer

        except Exception as e:
            raise RuntimeError(f"Failed to load tokenizer: {e}")

    @classmethod
    def load_model(cls, config: StepAudioModelConfig) -> "StepAudioModelWrapper":
        """
        Load or retrieve cached Step Audio TTS model.

        Args:
            config: Model configuration

        Returns:
            StepAudioModelWrapper instance
        """
        if StepAudioTTS is None:
            raise ImportError("StepAudioTTS not available. Step Audio not properly imported.")

        # Generate cache key
        cache_key = cls.get_cache_key(config)

        # CRITICAL: Auto-clear cache if config changed (dtype, quantization, flash attention, etc.)
        # This prevents memory leaks when switching model configurations
        if cls._last_config_key is not None and cls._last_config_key != cache_key:
            print(f"\n[StepAudio] ⚠️  Configuration changed!")
            print(f"[StepAudio]   Old config: {cls._last_config_key}")
            print(f"[StepAudio]   New config: {cache_key}")
            print(f"[StepAudio]   Clearing old model from VRAM...")

            # Use proper ComfyUI cleanup sequence
            cls.clear_cache(keep_tokenizer=True, force=True)

            # Extra wait for memory to be fully released (critical on Windows)
            import time
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                time.sleep(2)  # Wait 2 seconds for VRAM to be freed
                print(f"[StepAudio]   ✓ VRAM released, ready to load new model\n")

        # Check cache
        if cache_key in cls._model_cache:
            print(f"[StepAudio] Using cached model")
            cls._last_config_key = cache_key
            return cls._model_cache[cache_key]

        # Load model
        print(f"[StepAudio] Loading Step Audio model from {config.model_path}")
        print(f"[StepAudio] Config: quantization={config.quantization}, dtype={config.torch_dtype}, device={config.device_map}")
        print(f"[StepAudio] Attention Mechanism: {config.attention_mechanism}")

        # CRITICAL: Unload all ComfyUI models BEFORE loading new model (makes room in VRAM)
        try:
            import comfy.model_management as mm
            print(f"[StepAudio] Unloading other ComfyUI models to free VRAM...")
            mm.unload_all_models()
            mm.soft_empty_cache()
        except Exception as e:
            print(f"[StepAudio] Warning: Could not unload ComfyUI models: {e}")

        try:
            # Parse configurations
            torch_dtype = cls._parse_torch_dtype(config.torch_dtype)
            model_source = cls._parse_model_source(config.model_source)

            # Determine tokenizer path (usually in same directory or subdirectory)
            tokenizer_path = cls._find_tokenizer_path(config.model_path)

            # Load tokenizer first
            tokenizer = cls.load_tokenizer(
                encoder_path=tokenizer_path,
                model_source=config.model_source
            )

            # Prepare quantization config
            quantization_config = config.quantization if config.quantization != "none" else None

            # Load TTS model
            tts_model = StepAudioTTS(
                model_path=config.model_path,
                audio_tokenizer=tokenizer,
                model_source=model_source,
                tts_model_id=config.tts_model_id,
                quantization_config=quantization_config,
                torch_dtype=torch_dtype,
                device_map=config.device_map,
                attention_mechanism=config.attention_mechanism
            )

            # Wrap model
            wrapper = StepAudioModelWrapper(
                model=tts_model,
                tokenizer=tokenizer,
                config=config
            )

            # Cache model
            cls._model_cache[cache_key] = wrapper
            cls._last_config_key = cache_key  # Track current config
            print(f"[StepAudio] Model loaded and cached successfully")

            return wrapper

        except Exception as e:
            raise RuntimeError(f"Failed to load Step Audio model: {e}")

    @staticmethod
    def _find_tokenizer_path(model_path: str) -> str:
        """
        Find tokenizer path relative to model path.

        Args:
            model_path: Path to main model

        Returns:
            Path to tokenizer
        """
        # Common tokenizer locations
        possible_paths = [
            os.path.join(os.path.dirname(model_path), "Step-Audio-Tokenizer"),
            os.path.join(os.path.dirname(model_path), "tokenizer"),
            os.path.join(model_path, "tokenizer"),
            model_path,  # Sometimes tokenizer is in same dir
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path

        # Default to model path if nothing found
        print(f"[StepAudio] Warning: Tokenizer not found, using model path: {model_path}")
        return model_path

    @classmethod
    def clear_cache(cls, keep_tokenizer: bool = True, force: bool = False) -> None:
        """
        Clear model cache and free VRAM.

        NOTE: CUDA graphs are now disabled at model initialization (enable_cuda_graph=False),
        so we don't need complex graph cleanup logic.

        Args:
            keep_tokenizer: If True, keep tokenizer cached (recommended)
            force: If True, force VRAM clearing regardless of keep_in_vram setting
        """
        import gc
        import time

        # Get VRAM before clearing
        vram_before = cls.get_vram_stats()
        print(f"[StepAudio] Clearing cache...")

        # Synchronize
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Move to CPU
        for key, wrapper in cls._model_cache.items():
            try:
                if hasattr(wrapper, 'model'):
                    model = wrapper.model
                    if hasattr(model, 'llm') and hasattr(model.llm, 'to'):
                        model.llm.to('cpu')
                    if hasattr(model, 'cosy_model'):
                        if hasattr(model.cosy_model, 'model') and hasattr(model.cosy_model.model, 'to'):
                            model.cosy_model.model.to('cpu')
            except:
                pass

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            time.sleep(0.1)

        # Empty CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            time.sleep(0.2)

        # Clear references
        for key in list(cls._model_cache.keys()):
            try:
                wrapper = cls._model_cache[key]
                if hasattr(wrapper, 'model'):
                    wrapper.model = None
                if hasattr(wrapper, 'tokenizer'):
                    wrapper.tokenizer = None
                del wrapper
            except:
                pass

        cls._model_cache.clear()
        cls._last_config_key = None

        # Optionally clear tokenizer cache
        if not keep_tokenizer:
            for key in list(cls._tokenizer_cache.keys()):
                try:
                    del cls._tokenizer_cache[key]
                except:
                    pass
            cls._tokenizer_cache.clear()

        # ComfyUI cleanup
        try:
            import comfy.model_management as mm
            mm.unload_all_models()
            mm.soft_empty_cache(force=True)
        except:
            pass

        # Garbage collection
        for i in range(3):
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            time.sleep(0.1)

        # Final CUDA cleanup
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        # Report memory freed
        vram_after = cls.get_vram_stats()
        freed = vram_before['allocated'] - vram_after['allocated']
        print(f"[StepAudio] Freed {freed:.2f}GB VRAM")

        # Wait for memory to be fully released
        if torch.cuda.is_available():
            time.sleep(0.5)

    @classmethod
    def get_vram_stats(cls) -> Dict[str, float]:
        """
        Get VRAM usage statistics.

        Returns:
            Dictionary with VRAM stats in GB
        """
        if not torch.cuda.is_available():
            return {"allocated": 0.0, "reserved": 0.0, "total": 0.0}

        return {
            "allocated": torch.cuda.memory_allocated() / 1024**3,
            "reserved": torch.cuda.memory_reserved() / 1024**3,
            "total": torch.cuda.get_device_properties(0).total_memory / 1024**3,
        }


@dataclass
class StepAudioModelWrapper:
    """
    Wrapper for Step Audio TTS model and tokenizer.
    Provides convenient interface and VRAM management.
    """
    model: Any  # StepAudioTTS instance
    tokenizer: Any  # StepAudioTokenizer instance
    config: StepAudioModelConfig

    def clone(
        self,
        prompt_wav_path: str,
        prompt_text: str,
        target_text: str,
        temperature: float = 0.7,
        do_sample: bool = True,
        max_new_tokens: int = 8192,
        progress_bar=None
    ) -> Tuple[torch.Tensor, int]:
        """
        Perform voice cloning.

        Args:
            prompt_wav_path: Path to reference audio
            prompt_text: Text in reference audio
            target_text: Text to synthesize
            temperature: Sampling temperature (0.1-2.0)
            do_sample: Whether to use sampling
            max_new_tokens: Maximum number of new tokens to generate
            progress_bar: ComfyUI progress bar for token generation tracking

        Returns:
            (audio_tensor, sample_rate)
        """
        # CRITICAL FIX: Check if model is still valid (not cleared by cache management)
        if self.model is None:
            raise RuntimeError(
                "Model has been unloaded from memory. This can happen after VRAM clearing. "
                "Please reconnect the model node or reload the workflow."
            )

        # Pass parameters to the model
        return self.model.clone(
            prompt_wav_path,
            prompt_text,
            target_text,
            temperature,
            do_sample,
            max_new_tokens,
            progress_bar
        )

    def edit(
        self,
        input_audio_path: str,
        audio_text: str,
        edit_type: str,
        edit_info: Optional[str] = None,
        text: Optional[str] = None,
        n_edit_iter: int = 1,
        temperature: float = 0.7,  # Not used by StepAudioTTS.edit() - hardcoded to 0.7
        do_sample: bool = True,     # Not used by StepAudioTTS.edit() - hardcoded to True
        max_new_tokens: int = 8192  # Not used by StepAudioTTS.edit() - hardcoded to 8192
    ) -> Tuple[torch.Tensor, int]:
        """
        Perform audio editing with iterative refinement.

        Args:
            input_audio_path: Path to input audio
            audio_text: Transcript of input audio
            edit_type: Type of edit (emotion, style, speed, etc.)
            edit_info: Specific edit value
            text: Optional text for paralinguistic mode
            n_edit_iter: Number of edit iterations (1-5). Each iteration refines the edit.
            temperature: Sampling temperature (NOT USED - hardcoded to 0.7 in model)
            do_sample: Whether to use sampling (NOT USED - hardcoded to True in model)
            max_new_tokens: Maximum new tokens (NOT USED - hardcoded to 8192 in model)

        Returns:
            (audio_tensor, sample_rate)

        Note:
            The StepAudioTTS.edit() method has hardcoded generation parameters:
            - temperature=0.7, do_sample=True, max_new_tokens=8192
            These cannot be changed. The n_edit_iter parameter is implemented
            by calling edit() multiple times in sequence.
            Progress bar is not supported in edit mode.
        """
        import tempfile
        import soundfile as sf

        # CRITICAL FIX: Check if model is still valid (not cleared by cache management)
        if self.model is None:
            raise RuntimeError(
                "Model has been unloaded from memory. This can happen after VRAM clearing. "
                "Please reconnect the model node or reload the workflow."
            )

        # Clamp iterations to valid range
        n_edit_iter = max(1, min(5, n_edit_iter))

        # First iteration: edit the input audio
        print(f"[StepAudio]   Iteration 1/{n_edit_iter}...")
        audio_tensor, sample_rate = self.model.edit(
            input_audio_path=input_audio_path,
            audio_text=audio_text,
            edit_type=edit_type,
            edit_info=edit_info,
            text=text
        )

        # Additional iterations: refine the edit by re-editing the output
        if n_edit_iter > 1:
            for i in range(n_edit_iter - 1):
                print(f"[StepAudio]   Iteration {i+2}/{n_edit_iter}...")

                # Save current output to temp file
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    temp_path = tmp_file.name
                    # CRITICAL FIX: Use soundfile directly instead of torchaudio
                    # This bypasses torchaudio's backend selection issues in containers
                    # soundfile expects [samples, channels] for mono or [samples, channels] for stereo
                    if audio_tensor.dim() == 1:
                        waveform_np = audio_tensor.cpu().numpy()  # [samples]
                    else:
                        waveform_np = audio_tensor.cpu().numpy().T  # [channels, samples] -> [samples, channels]
                    sf.write(temp_path, waveform_np, sample_rate)

                try:
                    # Re-edit using the previous output
                    audio_tensor, sample_rate = self.model.edit(
                        input_audio_path=temp_path,
                        audio_text=audio_text,
                        edit_type=edit_type,
                        edit_info=edit_info,
                        text=text
                    )
                finally:
                    # Clean up temp file
                    import os
                    try:
                        os.unlink(temp_path)
                    except:
                        pass

        return audio_tensor, sample_rate

    def to_cpu(self) -> None:
        """
        Move model components to CPU with proper CUDA graph cleanup.

        CRITICAL: CUDA graphs MUST be cleared BEFORE moving to CPU!
        """
        import gc
        import time

        try:
            # Get offload device (usually CPU)
            try:
                import comfy.model_management as mm
                offload_device = mm.unet_offload_device()
            except:
                offload_device = torch.device('cpu')

            print(f"[StepAudio]   Offloading model to {offload_device}...")

            # ============================================
            # PHASE 1: CLEAR CUDA GRAPHS FIRST!
            # ============================================
            if torch.cuda.is_available():
                try:
                    # Synchronize before touching graphs
                    torch.cuda.synchronize()

                    # Clear CosyVoice CUDA graphs - try multiple possible structures
                    if hasattr(self.model, 'cosy_model'):
                        cosy = self.model.cosy_model

                        # Try path 1: cosy.cosy_impl.flow/hift structure
                        if hasattr(cosy, 'cosy_impl'):
                            cosy_impl = cosy.cosy_impl

                            # Clear decoder (flow) graphs
                            if hasattr(cosy_impl, 'flow'):
                                flow = cosy_impl.flow
                                if hasattr(flow, 'decoder'):
                                    if hasattr(flow.decoder, 'graph_chunk'):
                                        flow.decoder.graph_chunk = {}
                                    if hasattr(flow.decoder, 'use_cuda_graph'):
                                        flow.decoder.use_cuda_graph = False
                                if hasattr(flow, 'encoder'):
                                    if hasattr(flow.encoder, 'graph'):
                                        flow.encoder.graph = {}
                                    if hasattr(flow.encoder, 'inference_buffers'):
                                        flow.encoder.inference_buffers = {}
                                    if hasattr(flow.encoder, 'use_cuda_graph'):
                                        flow.encoder.use_cuda_graph = False

                            # Clear generator (hift) graphs
                            if hasattr(cosy_impl, 'hift'):
                                generator = cosy_impl.hift
                                if hasattr(generator, 'graph'):
                                    generator.graph = {}
                                if hasattr(generator, 'inference_buffers'):
                                    generator.inference_buffers = {}
                                if hasattr(generator, 'use_cuda_graph'):
                                    generator.use_cuda_graph = False

                        # Try path 2: cosy.model.decoder/generator structure
                        if hasattr(cosy, 'model'):
                            cosy_model = cosy.model

                            # Clear decoder graphs
                            if hasattr(cosy_model, 'decoder'):
                                decoder = cosy_model.decoder
                                if hasattr(decoder, 'graph_chunk'):
                                    decoder.graph_chunk = {}
                                if hasattr(decoder, 'use_cuda_graph'):
                                    decoder.use_cuda_graph = False

                            # Clear generator graphs
                            if hasattr(cosy_model, 'generator'):
                                generator = cosy_model.generator
                                if hasattr(generator, 'graph'):
                                    generator.graph = {}
                                if hasattr(generator, 'inference_buffers'):
                                    generator.inference_buffers = {}
                                if hasattr(generator, 'use_cuda_graph'):
                                    generator.use_cuda_graph = False

                            # Clear encoder graphs
                            if hasattr(cosy_model, 'encoder'):
                                encoder = cosy_model.encoder
                                if hasattr(encoder, 'graph'):
                                    encoder.graph = {}
                                if hasattr(encoder, 'inference_buffers'):
                                    encoder.inference_buffers = {}
                                if hasattr(encoder, 'use_cuda_graph'):
                                    encoder.use_cuda_graph = False

                        # Clear any top-level graph references
                        if hasattr(cosy, '_cuda_graphs'):
                            cosy._cuda_graphs = {}
                        if hasattr(cosy, 'cuda_graph'):
                            cosy.cuda_graph = None

                    # Synchronize and wait
                    torch.cuda.synchronize()
                    time.sleep(0.1)

                    print(f"[StepAudio]   ✓ CUDA graphs cleared")

                except Exception as e:
                    print(f"[StepAudio]   Warning: Could not clear CUDA graphs: {e}")

            # ============================================
            # PHASE 2: MOVE TO CPU
            # ============================================
            # Move LLM to CPU
            if hasattr(self.model, 'llm') and hasattr(self.model.llm, 'to'):
                self.model.llm.to(offload_device)
                print(f"[StepAudio]   ✓ LLM moved to {offload_device}")

            # Move CosyVoice components to CPU
            if hasattr(self.model, 'cosy_model'):
                if hasattr(self.model.cosy_model, 'model') and hasattr(self.model.cosy_model.model, 'to'):
                    self.model.cosy_model.model.to(offload_device)
                    print(f"[StepAudio]   ✓ CosyVoice model moved to {offload_device}")

                # Move frontend
                if hasattr(self.model.cosy_model, 'frontend') and hasattr(self.model.cosy_model.frontend, 'to'):
                    self.model.cosy_model.frontend.to(offload_device)

            # Fallback: try generic model path
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'to'):
                self.model.model.to(offload_device)

            # ============================================
            # PHASE 3: CLEANUP
            # ============================================
            # Garbage collection
            gc.collect()

            # Clear CUDA cache AFTER moving to CPU
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                print(f"[StepAudio]   ✓ CUDA cache cleared after offload")

        except Exception as e:
            print(f"[StepAudio]   ⚠️  Warning: Could not fully move model to CPU: {e}")

    def to_cuda(self, device: str = "cuda") -> None:
        """
        Move model components back to CUDA for inference.

        This follows ComfyUI's pattern of loading models to
        get_torch_device() when needed for inference.
        """
        try:
            # Get ComfyUI's preferred torch device
            try:
                import comfy.model_management as mm
                device = mm.get_torch_device()
            except:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            print(f"[StepAudio]   Loading model to {device}...")

            # Move LLM to CUDA
            if hasattr(self.model, 'llm'):
                if hasattr(self.model.llm, 'to'):
                    self.model.llm.to(device)
                    print(f"[StepAudio]   ✓ LLM loaded to {device}")

            # Move CosyVoice components to CUDA
            if hasattr(self.model, 'cosy_model'):
                if hasattr(self.model.cosy_model, 'model'):
                    if hasattr(self.model.cosy_model.model, 'to'):
                        self.model.cosy_model.model.to(device)
                        print(f"[StepAudio]   ✓ CosyVoice model loaded to {device}")

                # Move frontend
                if hasattr(self.model.cosy_model, 'frontend'):
                    if hasattr(self.model.cosy_model.frontend, 'to'):
                        self.model.cosy_model.frontend.to(device)

            # Fallback: try generic model path
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'to'):
                self.model.model.to(device)

        except Exception as e:
            print(f"[StepAudio]   ⚠️  Warning: Could not fully move model to CUDA: {e}")
