"""
Step Audio Audio Editing Node - Native ComfyUI Implementation
Edit audio with emotion, style, speed, paralinguistic effects, and denoising
"""

import os
import torch
from typing import Tuple, Dict, Any

from ..core import (
    StepAudioModelManager,
    StepAudioModelConfig,
    AudioEditor,
    discover_step_audio_models,
    get_step_audio_models_dir,
    load_text_resource,
    check_step_audio_installation,
    format_vram_usage
)


class StepAudioEditNode:
    """
    Native ComfyUI node for Step Audio audio editing.

    Features:
    - Edit emotion, style, speed, paralinguistic effects
    - Denoising and VAD
    - Iterative editing support
    - No JavaScript required - pure Python implementation
    """

    def __init__(self):
        self.model_wrapper = None
        self.current_config_key = None

    @classmethod
    def INPUT_TYPES(cls):
        """
        Define node inputs - only edit-specific parameters
        """
        # Discover available models
        discovered_models = discover_step_audio_models()
        model_options = discovered_models if discovered_models else ["(No models found - download Step-Audio-EditX)"]

        # Load edit options from resources
        emotions = load_text_resource("emotions.txt")
        styles = load_text_resource("styles.txt")
        paralinguistic = load_text_resource("paralinguistic.txt")

        # Build option lists
        emotion_options = ["none"] + emotions if emotions else ["none", "happy", "sad", "angry", "excited", "calm"]
        style_options = ["none"] + styles if styles else ["none", "whisper", "gentle", "serious", "casual"]
        speed_options = ["none", "faster", "slower", "more faster", "more slower"]
        para_options = ["none"] + paralinguistic if paralinguistic else ["none", "[Laughter]", "[Breathing]", "[Sigh]"]
        denoise_options = ["none", "denoise", "vad"]

        return {
            "required": {
                # 1. Audio text (transcript of input audio)
                "audio_text": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "dynamicPrompts": False,
                    "tooltip": "Exact transcript of what the input audio says. The model needs this to understand the content before editing the audio characteristics (emotion, style, speed, etc)."
                }),

                # 2-6. Model configuration
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

                # 7-12. Edit configuration
                "edit_type": (["emotion", "style", "speed", "paralinguistic", "denoising"], {
                    "default": "emotion",
                    "tooltip": "Edit category: emotion (happy, sad, angry), style (whisper, formal), speed (faster/slower), paralinguistic (laughter, breathing), denoising (clean audio). Only one type per edit."
                }),
                "emotion": (emotion_options, {
                    "default": "none",
                    "tooltip": "Target emotion (only if edit_type=emotion): happy, sad, angry, excited, calm, fearful, surprised, disgusted. Changes the emotional tone while keeping voice identity."
                }),
                "style": (style_options, {
                    "default": "none",
                    "tooltip": "Speaking style (only if edit_type=style): whisper, gentle, serious, casual, formal, friendly. Changes delivery style while keeping voice and emotion."
                }),
                "speed": (speed_options, {
                    "default": "none",
                    "tooltip": "Speed adjustment (only if edit_type=speed): faster (1.2x), slower (0.8x), more faster (1.5x), more slower (0.6x). Changes tempo without pitch shift."
                }),
                "paralinguistic": (para_options, {
                    "default": "none",
                    "tooltip": "Sound effect (only if edit_type=paralinguistic): [Laughter], [Breathing], [Sigh], [Gasp], [Cough]. Inserts natural non-speech sounds. If paralinguistic_text is empty, auto-appends effect to end of audio."
                }),
                "denoising": (denoise_options, {
                    "default": "none",
                    "tooltip": "Noise removal (only if edit_type=denoising): 'denoise' (remove background noise), 'vad' (voice activity detection, remove silence). Cleans up audio quality."
                }),

                # 13. Paralinguistic text (where to insert effect)
                "paralinguistic_text": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "dynamicPrompts": False,
                    "tooltip": "Text location for effect insertion (only if edit_type=paralinguistic). Example: 'I love this' - effect inserts before/after this phrase. Leave empty to auto-append effect to end of audio_text."
                }),

                # 14. Edit iterations
                "n_edit_iterations": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 5,
                    "step": 1,
                    "tooltip": "Edit strength through iteration: 1 (subtle change), 2-3 (moderate, recommended), 4-5 (strong, may degrade quality). Each iteration re-applies the edit to amplify the effect."
                }),

                # 15-17. Generation parameters
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Voice variation: 0.1-0.5 (consistent), 0.6-0.8 (natural, recommended), 0.9-1.5 (expressive), 1.6-2.0 (creative). Note: Edit mode hardcoded to 0.7, this parameter has no effect."
                }),
                "do_sample": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Sampling mode: True (natural speech) or False (deterministic, greedy decoding). Keep True. Note: Edit mode hardcoded to True, this parameter has no effect."
                }),
                "max_new_tokens": ("INT", {
                    "default": 8192,
                    "min": 256,
                    "max": 16384,
                    "step": 256,
                    "tooltip": "Maximum audio tokens to generate: 2048 (~10s), 4096 (~20s), 8192 (~40s). Higher = more VRAM + time. Note: Edit mode hardcoded to 8192, this parameter has no effect."
                }),

                # 18. Seed control
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "tooltip": "Reproducibility seed: 0 (random output each time) or any number (same input = same output). Use fixed seed to reproduce exact voice outputs."
                }),

                # 19. VRAM management
                "keep_model_in_vram": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Model caching: True (keep loaded, fast repeated use) or False (unload after use, frees ~8GB VRAM). Disable if switching between multiple models frequently."
                }),
            },
            "optional": {
                # Audio input to edit
                "input_audio": ("AUDIO", {
                    "tooltip": "Source audio to modify (0.5-30 seconds). The AI will apply the selected edit (emotion/style/speed/etc) while preserving voice identity and content."
                }),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "edit_audio"
    CATEGORY = "audio/step_audio"
    OUTPUT_NODE = False

    def edit_audio(
        self,
        audio_text: str,
        edit_type: str,
        emotion: str,
        style: str,
        speed: str,
        paralinguistic: str,
        denoising: str,
        paralinguistic_text: str,
        n_edit_iterations: int,
        model_path: str,
        device: str,
        torch_dtype: str,
        quantization: str,
        attention_mechanism: str,
        temperature: float,
        do_sample: bool,
        max_new_tokens: int,
        seed: int,
        keep_model_in_vram: bool,
        input_audio=None,
        **kwargs
    ) -> Tuple[Dict[str, Any]]:
        """
        Edit audio with specified effect.

        Args:
            audio_text: Transcript of the input audio
            edit_type: Type of edit to apply
            emotion/style/speed/paralinguistic/denoising: Edit parameters
            input_audio: Input audio (ComfyUI AUDIO dict)

        Returns:
            Tuple containing edited audio (ComfyUI AUDIO dict)
        """
        # Validate inputs
        if input_audio is None:
            raise ValueError("input_audio is required. Please connect an audio source.")

        if not audio_text.strip():
            raise ValueError("audio_text cannot be empty. Please provide the transcript of the input audio.")

        # Determine which edit_info to use based on edit_type
        edit_info_map = {
            "emotion": emotion,
            "style": style,
            "speed": speed,
            "paralinguistic": paralinguistic,
            "denoising": denoising
        }
        edit_info = edit_info_map.get(edit_type, "none")

        # Validate edit_info is not "none"
        if edit_info == "none":
            raise ValueError(
                f"Please select a valid {edit_type} option. "
                f"The current {edit_type} is set to 'none'."
            )

        # Auto-fill paralinguistic_text if empty when paralinguistic mode is selected
        if edit_type == "paralinguistic" and edit_info != "none" and not paralinguistic_text.strip():
            # Auto-append the paralinguistic effect to the end of audio_text
            paralinguistic_text = audio_text
            print(f"[StepAudio] Auto-appending '{edit_info}' to end of audio: '{audio_text[:50]}{'...' if len(audio_text) > 50 else ''}'")

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
        print(f"[StepAudio] Mode: AUDIO EDITING")
        print(f"[StepAudio] Edit Type: {edit_type.upper()}")
        print(f"[StepAudio] Edit Info: {edit_info}")
        print(f"[StepAudio] Iterations: {n_edit_iterations}")
        print(f"[StepAudio] VRAM: {format_vram_usage()}")
        print(f"[StepAudio] ═══════════════════════════════════\n")

        # Create audio editor and perform editing
        editor = AudioEditor(self.model_wrapper)
        output_audio = editor.edit_audio(
            input_audio=input_audio,
            audio_text=audio_text,
            edit_type=edit_type,
            edit_info=edit_info,
            paralinguistic_text=paralinguistic_text if edit_type == "paralinguistic" else None,
            n_edit_iterations=n_edit_iterations,
            seed=seed,
            temperature=temperature,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens
        )

        # VRAM management
        if not keep_model_in_vram:
            print(f"[StepAudio] 🧹 Clearing VRAM (keep_model_in_vram=False)...")
            StepAudioModelManager.clear_cache(keep_tokenizer=False)

            import time
            time.sleep(1.0)  # Give CUDA time to release memory

        print(f"\n[StepAudio] ═══════════════════════════════════")
        print(f"[StepAudio] Audio editing complete!")
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
