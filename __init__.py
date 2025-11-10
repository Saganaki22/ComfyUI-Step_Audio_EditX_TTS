"""
Step Audio EditX TTS - ComfyUI Node
Professional voice cloning and audio editing node powered by Step Audio EditX

Features:
- Voice Cloning: Zero-shot TTS from reference audio
- Audio Editing: Emotion, style, speed, paralinguistic effects
- Dual-codebook tokenizer (VQ02 + VQ06)
- 3B parameter LLM for audio generation
- 24kHz output with CosyVoice vocoder
"""

from .nodes import StepAudioCloneNode, StepAudioEditNode

# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "StepAudio_VoiceClone": StepAudioCloneNode,
    "StepAudio_AudioEdit": StepAudioEditNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StepAudio_VoiceClone": "StepAudioEditX - Clone 🎤",
    "StepAudio_AudioEdit": "StepAudioEditX - Edit ✏️"
}

# Web directory for custom JavaScript
#WEB_DIRECTORY = "./js"

# Version
__version__ = "1.0.1"

# Print banner on load
print("=" * 90)
print(f"[StepAudio] Loaded {len(NODE_CLASS_MAPPINGS)} nodes:")
print(f"[StepAudio]   • StepAudioEditX - Clone 🎤 - Native voice cloning node")
print(f"[StepAudio]   • StepAudioEditX - Edit ✏️ - Native audio editing node")
print(f"[StepAudio] Features: Voice Cloning | Audio Editing")
print(f"[StepAudio] Powered by: Step Audio EditX (3B) + CosyVoice")
print("=" * 90 + "\n")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
