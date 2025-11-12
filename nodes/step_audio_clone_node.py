"""
Step Audio Voice Cloning Node - Native ComfyUI Implementation
Zero-shot voice cloning from reference audio without JavaScript dependencies
"""

import os
import torch
from typing import Tuple, Dict, Any

from ..core import (
    StepAudioModelManager,
    StepAudioModelConfig,
    VoiceCloner,
    discover_step_audio_models,
    get_step_audio_models_dir,
    check_step_audio_installation,
    format_vram_usage
)


class StepAudioCloneNode:
    """
    Native ComfyUI node for Step Audio voice cloning.

    Features:
    - Zero-shot voice cloning from reference audio
    - 24kHz output with CosyVoice vocoder
    - No JavaScript required - pure Python implementation
    """

    def __init__(self):
        self.model_wrapper = None
        self.current_config_key = None

    @classmethod
    def INPUT_TYPES(cls):
        """
        Define node inputs - only clone-specific parameters
        """
        # Discover available models
        discovered_models = discover_step_audio_models()
        model_options = discovered_models if discovered_models else ["(No models found - download Step-Audio-EditX)"]

        return {
            "required": {
                # Text inputs
                "prompt_text": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "dynamicPrompts": False,
                    "tooltip": "Exact transcript of what the reference audio says. Must match the audio content perfectly for best voice cloning results. The AI uses this to align the voice characteristics."
                }),
                "target_text": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "dynamicPrompts": False,
                    "tooltip": "The new text you want to speak in the cloned voice. Can be any text - the AI will speak it with the reference voice's tone, accent, prosody, and speaking style."
                }),

                # Model configuration
                "model_path": (model_options, {
                    "default": model_options[0],
                    "tooltip": "Step-Audio-EditX model location. Place models in ComfyUI/models/step_audio/ or select from discovered models."
                }),
                "device": (["cuda", "cpu"], {
                    "default": "cuda",
                    "tooltip": "Hardware device: 'cuda' for GPU (10-100x faster, ~8GB VRAM) or 'cpu' (very slow, no VRAM needed). Always use CUDA if available."
                }),
                "torch_dtype": (["bfloat16", "float16", "float32", "auto"], {
                    "default": "bfloat16",
                    "tooltip": "Model precision: bfloat16 (best quality, stable, 8GB VRAM), float16 (good quality, 6GB VRAM), float32 (max quality, 16GB VRAM), auto (selects best for your GPU)."
                }),
                "quantization": (["none", "int4", "int8", "int4_awq"], {
                    "default": "none",
                    "tooltip": "VRAM reduction: 'none' (best quality, 8GB VRAM), int8 (good quality, 4GB VRAM), int4 (acceptable quality, 3GB VRAM), int4_awq (optimized int4, requires pre-quantized model). Use if low on VRAM."
                }),
                "attention_mechanism": (["sdpa", "eager", "flash_attn", "sage_attn"], {
                    "default": "sdpa",
                    "tooltip": "Attention layer: sdpa (fastest, good VRAM, default), eager (slowest, most stable), flash_attn (fastest, needs RTX 30xx+), sage_attn (best VRAM efficiency)."
                }),

                # Generation parameters
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Voice variation control: 0.1-0.5 (very consistent, robotic), 0.6-0.8 (natural, recommended), 0.9-1.5 (varied, expressive), 1.6-2.0 (very random, creative). Controls randomness in speech generation."
                }),
                "do_sample": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Sampling mode: True (natural, varied speech with temperature control) or False (deterministic, uses greedy decoding, ignores temperature). Keep True for natural-sounding voices."
                }),
                "max_new_tokens": ("INT", {
                    "default": 4096,
                    "min": 256,
                    "max": 16384,
                    "step": 256,
                    "tooltip": "Maximum audio tokens to generate: 2048 (~10s), 4096 (~20s), 8192 (~40s). Higher = more VRAM + time. With longform chunking enabled, each chunk is limited to this value."
                }),
                "longform_chunking": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Smart text splitting for long content (>2000 words). Splits at sentence boundaries, generates chunks separately, then stitches seamlessly. Slower but handles unlimited length. Disable for short text."
                }),

                # Seed control
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "tooltip": "Reproducibility seed: 0 (random output each time) or any number (same input = same output). Use fixed seed to reproduce exact voice outputs with identical parameters."
                }),

                # VRAM management
                "keep_model_in_vram": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Model caching: True (keep loaded in VRAM, fast repeated use) or False (unload after use, frees ~8GB VRAM). Disable if switching between multiple models frequently."
                }),
            },
            "optional": {
                # Audio input for reference voice
                "prompt_audio": ("AUDIO", {
                    "tooltip": "Reference voice audio (3-30 seconds recommended, 0.5-30s supported). The AI will analyze and clone this voice's tone, accent, pitch, rhythm, prosody, and speaking style for target_text generation."
                }),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "clone_voice"
    CATEGORY = "audio/step_audio"
    OUTPUT_NODE = False

    def clone_voice(
        self,
        prompt_text: str,
        target_text: str,
        model_path: str,
        device: str,
        torch_dtype: str,
        quantization: str,
        attention_mechanism: str,
        temperature: float,
        do_sample: bool,
        max_new_tokens: int,
        longform_chunking: bool,
        seed: int,
        keep_model_in_vram: bool,
        prompt_audio=None,
        **kwargs
    ) -> Tuple[Dict[str, Any]]:
        """
        Clone voice from reference audio.

        Args:
            prompt_text: Transcript of the reference audio
            target_text: Text to generate in the cloned voice
            prompt_audio: Reference audio (ComfyUI AUDIO dict)

        Returns:
            Tuple containing generated audio (ComfyUI AUDIO dict)
        """
        # Validate inputs
        if prompt_audio is None:
            raise ValueError("prompt_audio is required. Please connect an audio source.")

        if not prompt_text.strip():
            raise ValueError("prompt_text cannot be empty. Please provide the transcript of the reference audio.")

        if not target_text.strip():
            raise ValueError("target_text cannot be empty. Please provide the text to generate.")

        # Check Step Audio installation
        is_installed, error_msg = check_step_audio_installation()
        if not is_installed:
            raise RuntimeError(f"Step Audio not available: {error_msg}")

        # Resolve full model path
        full_model_path = self._resolve_model_path(model_path)
        if not full_model_path:
            raise ValueError(f"Model not found: {model_path}")

        # Load model if needed
        self._load_model_if_needed(
            model_path=full_model_path,
            quantization=quantization,
            torch_dtype=torch_dtype,
            device=device,
            attention_mechanism=attention_mechanism
        )

        print(f"\n[StepAudio] ═══════════════════════════════════")
        print(f"[StepAudio] Mode: VOICE CLONING")
        print(f"[StepAudio] Reference: {len(prompt_text)} chars")
        print(f"[StepAudio] Target: {len(target_text)} chars")
        print(f"[StepAudio] VRAM: {format_vram_usage()}")
        print(f"[StepAudio] ═══════════════════════════════════\n")

        # Create voice cloner and perform cloning
        cloner = VoiceCloner(self.model_wrapper)
        output_audio = cloner.clone_voice(
            prompt_audio=prompt_audio,
            prompt_text=prompt_text,
            target_text=target_text,
            seed=seed,
            temperature=temperature,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
            longform_chunking=longform_chunking
        )

        # VRAM management
        if not keep_model_in_vram:
            print(f"[StepAudio] 🧹 Clearing VRAM (keep_model_in_vram=False)...")
            StepAudioModelManager.clear_cache(keep_tokenizer=False)

            import time
            time.sleep(1.0)  # Give CUDA time to release memory

        print(f"\n[StepAudio] ═══════════════════════════════════")
        print(f"[StepAudio] Voice cloning complete!")
        print(f"[StepAudio] Final VRAM: {format_vram_usage()}")
        print(f"[StepAudio] ═══════════════════════════════════\n")

        return (output_audio,)

    def _resolve_model_path(self, model_name: str) -> str:
        """Resolve model name to full path"""
        # If already a full path, return as-is
        if os.path.isabs(model_name) and os.path.exists(model_name):
            return model_name

        # Try to find in models directory
        models_dir = get_step_audio_models_dir()
        if models_dir:
            full_path = os.path.join(models_dir, model_name)
            if os.path.exists(full_path):
                return full_path

        return model_name

    def _load_model_if_needed(
        self,
        model_path: str,
        quantization: str,
        torch_dtype: str,
        device: str,
        attention_mechanism: str
    ) -> None:
        """Load model if not already loaded or if config changed"""
        # Create config
        config = StepAudioModelConfig(
            model_path=model_path,
            model_source="local",  # Always use local models
            quantization=quantization if quantization != "none" else None,
            torch_dtype=torch_dtype,
            device_map=device,
            attention_mechanism=attention_mechanism
        )

        # Generate cache key
        cache_key = StepAudioModelManager.get_cache_key(config)

        # Check if already loaded with same config
        # CRITICAL: Also check if model inside wrapper is still valid (not cleared by cache)
        if (self.current_config_key == cache_key and
            self.model_wrapper is not None and
            hasattr(self.model_wrapper, 'model') and
            self.model_wrapper.model is not None):
            print(f"[StepAudio] Using existing model (cache hit)")
            return

        # If model was cleared but wrapper still exists, log it
        if self.model_wrapper is not None and hasattr(self.model_wrapper, 'model') and self.model_wrapper.model is None:
            print(f"[StepAudio] Model was unloaded from memory, reloading...")

        # Load new model
        print(f"[StepAudio] Loading model with new configuration...")
        self.model_wrapper = StepAudioModelManager.load_model(config)
        self.current_config_key = cache_key

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Force re-evaluation when inputs change"""
        return kwargs.get("seed", 0)
