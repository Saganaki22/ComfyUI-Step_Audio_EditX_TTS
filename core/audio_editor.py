"""
Audio Editing Logic for Step Audio EditX TTS
Handles edit mode orchestration
"""

import torch
from typing import Dict, Any, Tuple, Optional, List
from .model_manager import StepAudioModelWrapper
from .longform_chunker import create_chunker
from .audio_stitcher import create_stitcher
from .utils import (
    comfyui_audio_to_filepath,
    filepath_to_comfyui_audio,
    cleanup_temp_file,
    validate_audio_input,
    get_edit_type_options
)


class AudioEditor:
    """
    Handles audio editing operations using Step Audio EditX.
    """

    # Valid edit types
    VALID_EDIT_TYPES = ["emotion", "style", "speed", "paralinguistic", "denoising"]

    def __init__(self, model_wrapper: StepAudioModelWrapper):
        """
        Initialize audio editor.

        Args:
            model_wrapper: Loaded Step Audio model wrapper
        """
        self.model = model_wrapper

    def edit_audio(
        self,
        input_audio: Dict[str, Any],
        audio_text: str,
        edit_type: str,
        edit_info: Optional[str] = None,
        paralinguistic_text: Optional[str] = None,
        n_edit_iterations: int = 1,
        seed: int = 0,
        temperature: float = 0.7,
        do_sample: bool = True,
        max_new_tokens: int = 8192,
        longform_chunking: bool = False
    ) -> Dict[str, Any]:
        """
        Edit audio with specified modification.

        Args:
            input_audio: ComfyUI AUDIO format dict (audio to edit)
            audio_text: Transcript of input audio
            edit_type: Type of edit (emotion, style, speed, paralinguistic, denoising)
            edit_info: Specific edit value (e.g., 'happy', 'whisper', 'faster')
            paralinguistic_text: Text for paralinguistic mode (required if edit_type='paralinguistic')
            n_edit_iterations: Number of iterative edits (1-5)
            seed: Random seed (0 for random)

        Returns:
            ComfyUI AUDIO format dict with edited audio
        """
        # Validate inputs
        is_valid, error_msg = self.validate_inputs(
            input_audio, audio_text, edit_type, edit_info, paralinguistic_text
        )
        if not is_valid:
            raise ValueError(error_msg)

        # Set seed if specified
        if seed > 0:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        # Convert ComfyUI audio to temporary file
        print(f"[StepAudio] Converting input audio to file...")
        input_audio_path = comfyui_audio_to_filepath(input_audio, prefix="input")

        try:
            # Perform audio editing
            print(f"[StepAudio] Editing audio...")
            print(f"[StepAudio]   Audio text: {audio_text[:100]}{'...' if len(audio_text) > 100 else ''}")
            print(f"[StepAudio]   Edit type: {edit_type}")
            print(f"[StepAudio]   Edit info: {edit_info}")
            print(f"[StepAudio]   Iterations: {n_edit_iterations}")

            # Determine text parameter for Step Audio
            text_param = paralinguistic_text if edit_type == "paralinguistic" else None

            # Perform audio editing
            # Note: Chunking not implemented for edit mode (requires audio segmentation)
            audio_tensor, sample_rate = self._edit_single(
                input_audio_path=input_audio_path,
                audio_text=audio_text,
                edit_type=edit_type,
                edit_info=edit_info,
                text_param=text_param,
                n_edit_iterations=n_edit_iterations,
                temperature=temperature,
                do_sample=do_sample,
                max_new_tokens=max_new_tokens
            )

            print(f"[StepAudio] Audio editing completed successfully!")
            print(f"[StepAudio]   Output shape: {audio_tensor.shape}, Sample rate: {sample_rate}Hz")

            # Convert to ComfyUI audio format
            output_audio = self._tensor_to_comfyui_audio(audio_tensor, sample_rate)

            return output_audio

        finally:
            # Clean up temporary file
            cleanup_temp_file(input_audio_path)

    def _edit_single(
        self,
        input_audio_path: str,
        audio_text: str,
        edit_type: str,
        edit_info: Optional[str],
        text_param: Optional[str],
        n_edit_iterations: int,
        temperature: float,
        do_sample: bool,
        max_new_tokens: int
    ) -> Tuple[torch.Tensor, int]:
        """
        Perform single-shot audio editing (no chunking).

        Args:
            input_audio_path: Path to input audio file
            audio_text: Transcript of input audio
            edit_type: Type of edit
            edit_info: Specific edit value
            text_param: Text parameter for paralinguistic mode
            n_edit_iterations: Number of iterative edits
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            max_new_tokens: Maximum number of new tokens to generate

        Returns:
            (audio_tensor, sample_rate)
        """
        # Note: Progress bar not supported in edit mode (underlying StepAudioTTS.edit() doesn't support it)
        audio_tensor, sample_rate = self.model.edit(
            input_audio_path=input_audio_path,
            audio_text=audio_text,
            edit_type=edit_type,
            edit_info=edit_info,
            text=text_param,
            n_edit_iter=n_edit_iterations,
            temperature=temperature,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens
        )

        return audio_tensor, sample_rate

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

        return {
            "waveform": audio_tensor,
            "sample_rate": sample_rate
        }

    def validate_inputs(
        self,
        input_audio: Any,
        audio_text: str,
        edit_type: str,
        edit_info: Optional[str],
        paralinguistic_text: Optional[str]
    ) -> Tuple[bool, str]:
        """
        Validate edit inputs before processing.

        Args:
            input_audio: Input audio to edit
            audio_text: Transcript of input audio
            edit_type: Type of edit
            edit_info: Specific edit value
            paralinguistic_text: Text for paralinguistic mode

        Returns:
            (is_valid, error_message)
        """
        # Validate audio format
        if not validate_audio_input(input_audio):
            return False, "Invalid input_audio format"

        # Validate audio_text
        if not audio_text or not audio_text.strip():
            return False, "audio_text cannot be empty"

        # Validate edit_type
        if edit_type not in self.VALID_EDIT_TYPES:
            return False, f"Invalid edit_type '{edit_type}'. Must be one of: {', '.join(self.VALID_EDIT_TYPES)}"

        # Validate edit_info
        if edit_type != "denoising":  # Denoising doesn't always need edit_info
            if not edit_info or edit_info == "none":
                return False, f"edit_info required for edit_type '{edit_type}'"

            # Check if edit_info is valid for the edit_type
            valid_options = get_edit_type_options(edit_type)
            if valid_options and edit_info not in valid_options:
                return False, f"Invalid edit_info '{edit_info}' for edit_type '{edit_type}'. Valid options: {', '.join(valid_options[:5])}..."

        # Validate paralinguistic_text
        if edit_type == "paralinguistic":
            if not paralinguistic_text or not paralinguistic_text.strip():
                return False, "paralinguistic_text required when edit_type is 'paralinguistic'"

        # Check audio duration
        waveform = input_audio["waveform"]
        sample_rate = input_audio["sample_rate"]
        duration = waveform.shape[-1] / sample_rate

        if duration < 0.5:
            return False, f"Input audio too short ({duration:.2f}s). Minimum 0.5s required."

        if duration > 30.0:
            return False, f"Input audio too long ({duration:.2f}s). Maximum 30s supported."

        return True, ""

    @staticmethod
    def get_edit_options(edit_type: str) -> list:
        """
        Get valid options for a given edit type.

        Args:
            edit_type: Type of edit

        Returns:
            List of valid edit_info options
        """
        return get_edit_type_options(edit_type)
