"""
Voice Cloning Logic for Step Audio EditX TTS
Handles clone mode orchestration
"""

import torch
from typing import Dict, Any, Tuple, List
import comfy.utils
from .model_manager import StepAudioModelWrapper
from .longform_chunker import create_chunker
from .audio_stitcher import create_stitcher
from .utils import (
    comfyui_audio_to_filepath,
    filepath_to_comfyui_audio,
    cleanup_temp_file,
    validate_audio_input
)


class VoiceCloner:
    """
    Handles voice cloning operations using Step Audio EditX.
    """

    def __init__(self, model_wrapper: StepAudioModelWrapper):
        """
        Initialize voice cloner.

        Args:
            model_wrapper: Loaded Step Audio model wrapper
        """
        self.model = model_wrapper

    def clone_voice(
        self,
        prompt_audio: Dict[str, Any],
        prompt_text: str,
        target_text: str,
        trim_silence: bool = True,
        energy_norm: bool = True,
        seed: int = 0,
        temperature: float = 0.7,
        do_sample: bool = True,
        max_new_tokens: int = 8192,
        longform_chunking: bool = False
    ) -> Dict[str, Any]:
        """
        Clone a voice from reference audio and generate new speech.

        Args:
            prompt_audio: ComfyUI AUDIO format dict (reference voice)
            prompt_text: Text content of reference audio
            target_text: Text to synthesize in cloned voice
            trim_silence: Whether to trim silence from audio
            energy_norm: Whether to apply energy normalization
            seed: Random seed (0 for random)

        Returns:
            ComfyUI AUDIO format dict with generated speech
        """
        # Validate inputs
        if not validate_audio_input(prompt_audio):
            raise ValueError("Invalid prompt_audio format. Expected ComfyUI AUDIO dict.")

        if not prompt_text or not prompt_text.strip():
            raise ValueError("prompt_text cannot be empty")

        if not target_text or not target_text.strip():
            raise ValueError("target_text cannot be empty")

        # Set seed if specified
        if seed > 0:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        # Convert ComfyUI audio to temporary file
        print(f"[StepAudio] Converting reference audio to file...")
        prompt_audio_path = comfyui_audio_to_filepath(prompt_audio, prefix="prompt")

        try:
            # Perform voice cloning
            print(f"[StepAudio] Cloning voice...")
            print(f"[StepAudio]   Prompt text: {prompt_text[:100]}{'...' if len(prompt_text) > 100 else ''}")
            print(f"[StepAudio]   Target text: {target_text[:100]}{'...' if len(target_text) > 100 else ''}")

            # Check if chunking is needed
            if longform_chunking:
                print(f"[StepAudio] 📊 Longform chunking enabled")

                # Count text tokens accurately
                try:
                    text_tokenizer = self.model.model.tokenizer
                    target_tokens = len(text_tokenizer.encode(target_text))
                    print(f"[StepAudio]    Using accurate tokenizer for counting")
                except Exception:
                    # Fallback: character-based estimate
                    target_tokens = len(target_text) // 4
                    print(f"[StepAudio]    Using character-based token estimate")

                # For TTS: text tokens → audio tokens ratio is HIGHLY VARIABLE (5-15x)
                # We use a conservative estimate to ensure chunking happens when needed
                # Real-world observation: 150 text tokens can need 1000+ audio tokens
                AUDIO_TOKEN_RATIO = 12  # Conservative: 1 text token → 12 audio tokens (was 5, too low!)
                estimated_audio_tokens = target_tokens * AUDIO_TOKEN_RATIO

                print(f"[StepAudio]    Target text tokens: {target_tokens}")
                print(f"[StepAudio]    Estimated audio output: ~{estimated_audio_tokens} tokens (conservative estimate)")
                print(f"[StepAudio]    Max new tokens limit: {max_new_tokens}")
                print(f"[StepAudio]    Safety margin: Using {AUDIO_TOKEN_RATIO}x ratio to ensure proper chunking")

                # Check if we need to chunk based on estimated audio output
                if estimated_audio_tokens > max_new_tokens:
                    print(f"[StepAudio]    ⚠️  Estimated output ({estimated_audio_tokens}) exceeds limit ({max_new_tokens})")
                    print(f"[StepAudio]    ✓ Chunking target text into smaller pieces")

                    # Calculate chunk size in text tokens
                    # Each text token generates ~12 audio tokens, so divide max_new_tokens by ratio
                    # Use 80% safety margin to ensure we don't exceed max_new_tokens
                    chunk_size_text_tokens = max(50, int(max_new_tokens / AUDIO_TOKEN_RATIO * 0.8))
                    print(f"[StepAudio]    Chunk size: {chunk_size_text_tokens} text tokens (~{chunk_size_text_tokens * AUDIO_TOKEN_RATIO} audio tokens)")

                    # Create chunker with calculated size
                    try:
                        text_tokenizer = self.model.model.tokenizer
                    except:
                        text_tokenizer = None

                    chunker = create_chunker(
                        max_tokens=chunk_size_text_tokens,
                        tokenizer=text_tokenizer,
                        enforce_minimum=False  # Allow chunks smaller than MIN_CHUNK_TOKENS for small max_new_tokens
                    )

                    # Perform chunked generation
                    audio_tensor, sample_rate = self._clone_with_chunking(
                        prompt_audio_path=prompt_audio_path,
                        prompt_text=prompt_text,
                        target_text=target_text,
                        chunker=chunker,
                        temperature=temperature,
                        do_sample=do_sample,
                        max_new_tokens=max_new_tokens
                    )
                else:
                    print(f"[StepAudio]    ℹ️  Estimated output fits within limit, single-shot generation")
                    # Generate in one go
                    audio_tensor, sample_rate = self._clone_single(
                        prompt_audio_path=prompt_audio_path,
                        prompt_text=prompt_text,
                        target_text=target_text,
                        temperature=temperature,
                        do_sample=do_sample,
                        max_new_tokens=max_new_tokens
                    )
            else:
                # Standard single-shot generation (no chunking)
                # Note: If text is too long, generation will be truncated at max_new_tokens
                print(f"[StepAudio] Single-shot generation (max_new_tokens: {max_new_tokens})")
                audio_tensor, sample_rate = self._clone_single(
                    prompt_audio_path=prompt_audio_path,
                    prompt_text=prompt_text,
                    target_text=target_text,
                    temperature=temperature,
                    do_sample=do_sample,
                    max_new_tokens=max_new_tokens
                )

            print(f"[StepAudio] Voice cloning completed successfully!")
            print(f"[StepAudio]   Output shape: {audio_tensor.shape}, Sample rate: {sample_rate}Hz")

            # Convert to ComfyUI audio format
            output_audio = self._tensor_to_comfyui_audio(audio_tensor, sample_rate)

            return output_audio

        finally:
            # Clean up temporary file
            cleanup_temp_file(prompt_audio_path)

    def _clone_single(
        self,
        prompt_audio_path: str,
        prompt_text: str,
        target_text: str,
        temperature: float,
        do_sample: bool,
        max_new_tokens: int
    ) -> Tuple[torch.Tensor, int]:
        """
        Perform single-shot voice cloning (no chunking).

        Args:
            prompt_audio_path: Path to reference audio
            prompt_text: Text content of reference audio
            target_text: Text to synthesize
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            max_new_tokens: Maximum number of new tokens to generate

        Returns:
            (audio_tensor, sample_rate)
        """
        # Create ComfyUI progress bar for token generation tracking
        pbar = comfy.utils.ProgressBar(max_new_tokens)

        audio_tensor, sample_rate = self.model.clone(
            prompt_wav_path=prompt_audio_path,
            prompt_text=prompt_text,
            target_text=target_text,
            temperature=temperature,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
            progress_bar=pbar
        )

        return audio_tensor, sample_rate

    def _clone_with_chunking(
        self,
        prompt_audio_path: str,
        prompt_text: str,
        target_text: str,
        chunker,
        temperature: float,
        do_sample: bool,
        max_new_tokens: int
    ) -> Tuple[torch.Tensor, int]:
        """
        Perform chunked voice cloning for long-form text.

        Args:
            prompt_audio_path: Path to reference audio
            prompt_text: Text content of reference audio
            target_text: Text to synthesize (will be chunked)
            chunker: LongformChunker instance
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            max_new_tokens: Maximum new tokens to generate per chunk

        Returns:
            (stitched_audio_tensor, sample_rate)
        """
        # Chunk the target text
        chunks = chunker.chunk_text(target_text)
        num_chunks = len(chunks)

        print(f"[StepAudio] 🎯 Generating {num_chunks} chunks...")

        # Generate audio for each chunk
        audio_chunks = []
        sample_rate = None

        for i, (chunk_text, start_char, end_char) in enumerate(chunks):
            chunk_info = chunker.get_chunk_info(i, num_chunks)
            print(f"\n[StepAudio] 📝 {chunk_info}: {len(chunk_text)} chars")
            print(f"[StepAudio]    Text: {chunk_text[:80]}{'...' if len(chunk_text) > 80 else ''}")

            # Create progress bar for this chunk
            # Progress bar tracks audio token generation, not text tokens
            pbar = comfy.utils.ProgressBar(max_new_tokens)

            # Generate audio for chunk
            chunk_audio, chunk_sr = self.model.clone(
                prompt_wav_path=prompt_audio_path,
                prompt_text=prompt_text,
                target_text=chunk_text,
                temperature=temperature,
                do_sample=do_sample,
                max_new_tokens=max_new_tokens,
                progress_bar=pbar
            )

            audio_chunks.append(chunk_audio)

            # Store sample rate from first chunk
            if sample_rate is None:
                sample_rate = chunk_sr

            print(f"[StepAudio]    ✓ Generated: {chunk_audio.shape[0]} samples ({chunk_audio.shape[0] / chunk_sr:.2f}s)")

        # Stitch chunks together
        print(f"\n[StepAudio] 🔗 Stitching {num_chunks} audio chunks...")
        stitcher = create_stitcher(sample_rate=sample_rate, crossfade_ms=0)  # No crossfade - simple concatenation
        stitched_audio, final_sr = stitcher.stitch_chunks(audio_chunks, sample_rate)

        return stitched_audio, final_sr

    @staticmethod
    def _tensor_to_comfyui_audio(audio_tensor: torch.Tensor, sample_rate: int) -> Dict[str, Any]:
        """
        Convert audio tensor to ComfyUI AUDIO format.

        Args:
            audio_tensor: Audio tensor from Step Audio (1D or 2D)
            sample_rate: Sample rate in Hz

        Returns:
            ComfyUI AUDIO format dict
        """
        # Ensure tensor is 3D: [batch, channels, samples]
        if audio_tensor.dim() == 1:
            # [samples] -> [1, 1, samples]
            audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)
        elif audio_tensor.dim() == 2:
            # [channels, samples] -> [1, channels, samples]
            audio_tensor = audio_tensor.unsqueeze(0)

        # Ensure channels is first (after batch)
        # Step Audio typically outputs [samples] or [1, samples]
        # We want [1, channels, samples] where channels=1 for mono

        return {
            "waveform": audio_tensor,
            "sample_rate": sample_rate
        }

    def validate_inputs(
        self,
        prompt_audio: Any,
        prompt_text: str,
        target_text: str
    ) -> Tuple[bool, str]:
        """
        Validate clone inputs before processing.

        Args:
            prompt_audio: Reference audio
            prompt_text: Text in reference audio
            target_text: Text to synthesize

        Returns:
            (is_valid, error_message)
        """
        if not validate_audio_input(prompt_audio):
            return False, "Invalid prompt_audio format"

        if not prompt_text or not prompt_text.strip():
            return False, "prompt_text cannot be empty"

        if not target_text or not target_text.strip():
            return False, "target_text cannot be empty"

        # Check audio duration (Step Audio has limits)
        waveform = prompt_audio["waveform"]
        sample_rate = prompt_audio["sample_rate"]
        duration = waveform.shape[-1] / sample_rate

        if duration < 0.5:
            return False, f"Reference audio too short ({duration:.2f}s). Minimum 0.5s required."

        if duration > 30.0:
            return False, f"Reference audio too long ({duration:.2f}s). Maximum 30s supported."

        return True, ""
