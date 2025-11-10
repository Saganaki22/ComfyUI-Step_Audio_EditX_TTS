"""
Utility functions for Step Audio EditX TTS ComfyUI Node
Handles audio I/O, format conversion, and resource loading
"""

import os
import sys
import json
import tempfile
import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any


def get_step_audio_models_dir() -> Optional[str]:
    """
    Get the Step Audio models directory from ComfyUI's models folder.
    Searches in multiple potential locations.

    Returns:
        Path to models directory or None if not found
    """
    try:
        import folder_paths

        # FIXED: Search paths in priority order - now looks for Step-Audio-EditX
        search_paths = [
            os.path.join(folder_paths.models_dir, "Step-Audio-EditX"),
            os.path.join(folder_paths.models_dir, "checkpoints", "Step-Audio-EditX"),
            os.path.join(folder_paths.models_dir, "StepAudioEditX"),
            os.path.join(folder_paths.models_dir, "step-audio-editx"),
        ]

        for path in search_paths:
            if os.path.exists(path) and os.path.isdir(path):
                return path

        return None
    except Exception as e:
        print(f"[StepAudio] Error finding models directory: {e}")
        return None


def discover_step_audio_models() -> List[str]:
    """
    Auto-discover Step Audio models in the models directory.
    Looks for directories containing config.json or model files.

    Returns:
        List of model names/paths
    """
    models_dir = get_step_audio_models_dir()
    if not models_dir:
        return []

    discovered_models = []

    try:
        for item in os.listdir(models_dir):
            item_path = os.path.join(models_dir, item)
            if os.path.isdir(item_path):
                # Check for config.json or model files
                has_config = os.path.exists(os.path.join(item_path, "config.json"))
                has_model = any(
                    os.path.exists(os.path.join(item_path, f))
                    for f in ["model.safetensors", "pytorch_model.bin", "model.bin"]
                )

                if has_config or has_model:
                    discovered_models.append(item)

        return sorted(discovered_models)
    except Exception as e:
        print(f"[StepAudio] Error discovering models: {e}")
        return []


def load_text_resource(filename: str) -> List[str]:
    """
    Load a text resource file (emotions, styles, etc.)

    Args:
        filename: Name of the resource file (e.g., 'emotions.txt')

    Returns:
        List of strings (one per line, comments and empty lines removed)
    """
    try:
        resources_dir = Path(__file__).parent.parent / "resources"
        filepath = resources_dir / filename

        if not filepath.exists():
            print(f"[StepAudio] Warning: Resource file not found: {filepath}")
            return []

        with open(filepath, 'r', encoding='utf-8') as f:
            lines = [
                line.strip()
                for line in f.readlines()
                if line.strip() and not line.strip().startswith('#')
            ]

        return lines
    except Exception as e:
        print(f"[StepAudio] Error loading resource {filename}: {e}")
        return []


def load_json_resource(filename: str) -> Dict:
    """
    Load a JSON resource file.

    Args:
        filename: Name of the JSON file (e.g., 'clone_presets.json')

    Returns:
        Dictionary with JSON contents
    """
    try:
        resources_dir = Path(__file__).parent.parent / "resources"
        filepath = resources_dir / filename

        if not filepath.exists():
            print(f"[StepAudio] Warning: JSON resource not found: {filepath}")
            return {}

        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"[StepAudio] Error loading JSON resource {filename}: {e}")
        return {}


def comfyui_audio_to_filepath(audio_data: Dict[str, Any], prefix: str = "step_audio") -> str:
    """
    Convert ComfyUI AUDIO format to a temporary WAV file.

    Args:
        audio_data: Dictionary with 'waveform' (torch.Tensor) and 'sample_rate' (int)
        prefix: Prefix for temporary file name

    Returns:
        Path to temporary WAV file (caller must clean up)
    """
    try:
        waveform = audio_data["waveform"]  # [batch, channels, samples]
        sample_rate = audio_data["sample_rate"]

        # Ensure waveform is 2D [channels, samples]
        if waveform.dim() == 3:
            waveform = waveform.squeeze(0)  # Remove batch dimension
        elif waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # Add channel dimension

        # Convert to CPU if needed
        if waveform.is_cuda:
            waveform = waveform.cpu()

        # Create temporary file
        temp_fd, temp_path = tempfile.mkstemp(suffix=".wav", prefix=f"{prefix}_")
        os.close(temp_fd)

        # Save as WAV
        torchaudio.save(temp_path, waveform, sample_rate)

        return temp_path

    except Exception as e:
        raise RuntimeError(f"Failed to convert ComfyUI audio to file: {e}")


def filepath_to_comfyui_audio(filepath: str) -> Dict[str, Any]:
    """
    Load a WAV file and convert to ComfyUI AUDIO format.

    Args:
        filepath: Path to WAV file

    Returns:
        Dictionary with 'waveform' (torch.Tensor [1, channels, samples]) and 'sample_rate' (int)
    """
    try:
        waveform, sample_rate = torchaudio.load(filepath)

        # Ensure waveform is 3D [batch, channels, samples]
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)  # Add batch dimension
        elif waveform.dim() == 1:
            waveform = waveform.unsqueeze(0).unsqueeze(0)  # Add batch and channel

        return {
            "waveform": waveform,
            "sample_rate": sample_rate
        }

    except Exception as e:
        raise RuntimeError(f"Failed to load audio file {filepath}: {e}")


def cleanup_temp_file(filepath: str) -> None:
    """
    Safely delete a temporary file.

    Args:
        filepath: Path to file to delete
    """
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
    except Exception as e:
        print(f"[StepAudio] Warning: Failed to clean up temp file {filepath}: {e}")


def get_edit_type_options(edit_type: str) -> List[str]:
    """
    Get the available options for a given edit type.

    Args:
        edit_type: One of 'emotion', 'style', 'speed', 'paralinguistic', 'denoising'

    Returns:
        List of valid options for that edit type
    """
    if edit_type == "emotion":
        return load_text_resource("emotions.txt")
    elif edit_type == "style":
        return load_text_resource("styles.txt")
    elif edit_type == "speed":
        return ["faster", "slower", "more faster", "more slower"]
    elif edit_type == "paralinguistic":
        return load_text_resource("paralinguistic.txt")
    elif edit_type == "denoising":
        return ["denoise", "vad"]
    else:
        return []


def check_step_audio_installation() -> Tuple[bool, str]:
    """
    Check if Step Audio dependencies are installed (now bundled).

    Returns:
        (is_installed, error_message)
    """
    try:
        # Check bundled step_audio_impl directory
        impl_dir = Path(__file__).parent.parent / "step_audio_impl"
        if not impl_dir.exists():
            return False, "step_audio_impl directory not found. Please ensure it was copied correctly."

        # Try importing key modules from bundled implementation
        if str(impl_dir) not in sys.path:
            sys.path.insert(0, str(impl_dir))

        from tts import StepAudioTTS
        from tokenizer import StepAudioTokenizer
        from model_loader import model_loader

        return True, ""
    except ImportError as e:
        return False, f"Step Audio bundled implementation has import errors: {e}"
    except Exception as e:
        return False, f"Error checking Step Audio installation: {e}"


def get_bundled_impl_path() -> Path:
    """
    Get path to bundled Step Audio implementation.

    Returns:
        Path to step_audio_impl directory
    """
    return Path(__file__).parent.parent / "step_audio_impl"


def tensor_to_numpy(audio_tensor: torch.Tensor) -> np.ndarray:
    """
    Convert audio tensor to numpy array.

    Args:
        audio_tensor: Torch tensor (any shape)

    Returns:
        Numpy array
    """
    if audio_tensor.is_cuda:
        audio_tensor = audio_tensor.cpu()
    return audio_tensor.numpy()


def numpy_to_tensor(audio_array: np.ndarray, device: str = "cpu") -> torch.Tensor:
    """
    Convert numpy array to torch tensor.

    Args:
        audio_array: Numpy array
        device: Target device ('cpu' or 'cuda')

    Returns:
        Torch tensor
    """
    tensor = torch.from_numpy(audio_array)
    if device == "cuda" and torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor


def validate_audio_input(audio_data: Any) -> bool:
    """
    Validate that audio_data is in correct ComfyUI AUDIO format.

    Args:
        audio_data: Audio data to validate

    Returns:
        True if valid, False otherwise
    """
    if not isinstance(audio_data, dict):
        return False

    if "waveform" not in audio_data or "sample_rate" not in audio_data:
        return False

    if not isinstance(audio_data["waveform"], torch.Tensor):
        return False

    if not isinstance(audio_data["sample_rate"], int):
        return False

    return True


def format_vram_usage() -> str:
    """
    Get formatted VRAM usage string.

    Returns:
        String like "8.2GB / 24.0GB" or "N/A" if CUDA unavailable
    """
    if not torch.cuda.is_available():
        return "N/A (CPU)"

    try:
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3  # GB
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB

        return f"{allocated:.1f}GB / {total:.1f}GB (Reserved: {reserved:.1f}GB)"
    except Exception:
        return "N/A"


def clear_cuda_cache() -> None:
    """
    Clear CUDA cache and run garbage collection.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    import gc
    gc.collect()
