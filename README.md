<h1 align="center">ComfyUI Step Audio EditX TTS</h1>

<p align="center">
  <img src="step_audio_impl/assets/logo.png"  height=100>
</p>

<div align="center">
    <a href="https://stepaudiollm.github.io/step-audio-editx/"><img src="https://img.shields.io/static/v1?label=Demo%20Page&message=Web&color=green"></a> &ensp;
    <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.12+-blue.svg"></a> &ensp;
    <a href="https://arxiv.org/abs/2511.03601"><img src="https://img.shields.io/static/v1?label=Tech%20Report&message=Arxiv&color=red"></a> &ensp;
    <a href="https://huggingface.co/stepfun-ai/Step-Audio-EditX"><img src="https://img.shields.io/static/v1?label=Step-Audio-EditX&message=HuggingFace&color=yellow"></a> &ensp;
    <a href="https://github.com/comfyanonymous/ComfyUI"><img src="https://img.shields.io/badge/ComfyUI-Native-green.svg"></a> &ensp;
    <a href="https://huggingface.co/spaces/stepfun-ai/Step-Audio-EditX"><img src="https://img.shields.io/static/v1?label=Space%20Playground&message=HuggingFace&color=yellow"></a> &ensp;
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg"></a>
</div>

<img width="1505" height="1032" alt="image" src="https://github.com/user-attachments/assets/42c63b81-98e0-49ec-9199-3fbb04e2db9c" />


Professional native ComfyUI nodes for **Step Audio EditX** - State-of-the-art zero-shot voice cloning and audio editing with emotion, style, speed control, and more.

## 🎯 Key Features

- **🎤 Zero-Shot Voice Cloning**: Clone any voice from just 3-30 seconds of reference audio
- **🎭 Advanced Audio Editing**: Edit emotion, speaking style, speed, add paralinguistic effects, and denoise
- **⚡ Native ComfyUI Integration**: Pure Python implementation - no JavaScript required
- **🧩 Modular Workflow Design**: Separate nodes for cloning and editing workflows
- **🎛️ Advanced Controls**: Full model configuration, generation parameters, and VRAM management
- **📊 Longform Support**: Smart chunking for unlimited text length with seamless stitching
- **🔄 Iterative Editing**: Multi-iteration editing for stronger, more pronounced effects

---

## Demos

<table>
  <tr>
    <th style="vertical-align : middle;text-align: center">Task</th>
    <th style="vertical-align : middle;text-align: center">Text</th>
    <th style="vertical-align : middle;text-align: center">Source</th>
    <th style="vertical-align : middle;text-align: center">Edited</th>
  </tr>

  <tr>
    <td align="center"> Emotion-Fear</td>
    <td align="center"> 我总觉得，有人在跟着我，我能听到奇怪的脚步声。</td>
    <td align="center">

  [fear_zh_female_prompt.webm](https://github.com/user-attachments/assets/a088c059-032c-423f-81d6-3816ba347ff5) 
  </td>
    <td align="center">
      
  [fear_zh_female_output.webm](https://github.com/user-attachments/assets/917494ac-5913-4949-8022-46cf55ca05dd)
  </td>
  </tr>


  <tr>
    <td align="center"> Style-Whisper</td>
    <td align="center"> 比如在工作间隙，做一些简单的伸展运动，放松一下身体，这样，会让你更有精力。</td>
    <td align="center">
      
  [whisper_prompt.webm](https://github.com/user-attachments/assets/ed9e22f1-1bac-417b-913a-5f1db31f35c9)
  </td>
    <td align="center">
      
  [whisper_output.webm](https://github.com/user-attachments/assets/e0501050-40db-4d45-b380-8bcc309f0b5f)
  </td>
  </tr>

  <tr>
    <td align="center"> Style-Act_coy</td>
    <td align="center"> 我今天想喝奶茶，可是不知道喝什么口味，你帮我选一下嘛，你选的都好喝～</td>
    <td align="center">

  [act_coy_prompt.webm](https://github.com/user-attachments/assets/74d60625-5b3c-4f45-becb-0d3fe7cc4b3f)
  </td>
    <td align="center"> 

  [act_coy_output.webm](https://github.com/user-attachments/assets/b2f74577-56c2-4997-afd6-6bf47d15ea51)
  </td>
  </tr>


  <tr>
    <td align="center"> Paralinguistics</td>
    <td align="center"> 你这次又忘记带钥匙了 [Dissatisfaction-hnn]，真是拿你没办法。</td>
    <td align="center">
      
  [paralingustic_prompt.webm](https://github.com/user-attachments/assets/21e831a3-8110-4c64-a157-60e0cf6735f0)
  </td>
    <td align="center">
      
  [paralingustic_output.webm](https://github.com/user-attachments/assets/a82f5a40-c6a3-409b-bbe6-271180b20d7b)
  </td>
  </tr>


  <tr>
    <td align="center"> Denoising</td>
    <td align="center"> Such legislation was clarified and extended from time to time thereafter. No, the man was not drunk, he wondered how we got tied up with this stranger. Suddenly, my reflexes had gone. It's healthier to cook without sugar.</td>
    <td align="center">
      
  [denoising_prompt.webm](https://github.com/user-attachments/assets/70464bf4-ebde-44a3-b2a6-8c292333319b)
  </td>
    <td align="center">
      
  [denoising_output.webm](https://github.com/user-attachments/assets/7cd0ae8d-1bf0-40fc-9bcd-f419bd4b2d21)
  </td>
  </tr>

  <tr>
    <td align="center"> Speed-Faster</td>
    <td align="center"> 上次你说鞋子有点磨脚，我给你买了一双软软的鞋垫。</td>
    <td align="center">
      
  [speed_faster_prompt.webm](https://github.com/user-attachments/assets/db46609e-1b98-48d8-99c8-e166cfdfc6e3)
  </td>
    <td align="center">
      
  [speed_faster_output.webm](https://github.com/user-attachments/assets/0fbc14ca-dd4a-4362-aadc-afe0629f4c9f)
  </td>
  </tr>
  
</table>

---

## 📦 Installation

### Method 1: ComfyUI Manager (Recommended)

1. Open ComfyUI Manager
2. Search for "Step Audio EditX TTS"
3. Click Install
4. Restart ComfyUI

### Method 2: Manual Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Saganaki22/ComfyUI-Step_Audio_EditX_TTS
cd ComfyUI-Step_Audio_TTS
pip install -r requirements.txt
```

### 📥 Download Models

Download **both** the Step-Audio-EditX model and Step-Audio-Tokenizer. They must be in the correct folder structure:

```bash
cd ComfyUI/models
mkdir -p Step-Audio-EditX
cd Step-Audio-EditX

# Download BOTH repositories from HuggingFace
git clone https://huggingface.co/stepfun-ai/Step-Audio-EditX
git clone https://huggingface.co/stepfun-ai/Step-Audio-Tokenizer
```

**Required Folder Structure:**
```
ComfyUI/
└── models/
    └── Step-Audio-EditX/
        ├── Step-Audio-EditX/       ← Main model
        └── Step-Audio-Tokenizer/   ← Tokenizer (required)
```

Or download manually from:
- https://huggingface.co/stepfun-ai/Step-Audio-EditX
- https://huggingface.co/stepfun-ai/Step-Audio-Tokenizer

---

## 💻 VRAM Requirements

| Node | Precision | Quantization | VRAM Usage |
|------|-----------|--------------|------------|
| Clone | bfloat16 | none | ~11-14GB |
| Edit | bfloat16 | none | ~14-18GB |
| Clone | float16 | int8 | ~9-11GB |
| Edit | float16 | int8 | ~10-12GB |

**Note**: Edit node uses more VRAM than clone node. We are waiting for optimized quantized models from the Step AI research team and will implement them for lower VRAM usage when available.

**Recommendations**:
- **RTX 4090/A6000+**: Use bfloat16 + no quantization for best quality
- **RTX 3090/4080**: Use bfloat16 + int8 quantization
- **Lower VRAM**: Use int4 quantization (quality trade-off)

---

## 🎤 Clone Node

### Overview

Zero-shot voice cloning from reference audio. The AI analyzes a 3-30 second voice sample and generates new speech in that voice with any text you provide.

**Use Cases**:
- Character voice generation for games and animations
- Narration and professional voiceovers
- Voice consistency across long-form content
- Multilingual voice cloning

### Parameters

#### Text Inputs
- **prompt_text**: Exact transcript of reference audio (must match perfectly for best results)
- **target_text**: New text to speak in cloned voice

#### Model Configuration
- **model_path**: Path to Step-Audio-EditX model
- **device**: `cuda` (GPU, fast) or `cpu` (slow, fallback)
- **torch_dtype**: `bfloat16` (best), `float16` (good), `float32` (max quality), `auto`
- **quantization**: `none` (best), `int8` (balanced), `int4` (lower VRAM), `int4_awq` (optimized int4)
- **attention_mechanism**: `sdpa` (default), `eager`, `flash_attn`, `sage_attn`

#### Generation Parameters
- **temperature**: Voice variation (0.6-0.8 recommended for natural speech)
- **do_sample**: Keep `True` for natural-sounding voices
- **max_new_tokens**: Audio tokens to generate (4096 ≈ 20s, 8192 ≈ 40s)
- **longform_chunking**: Enable for text >2000 words (auto-splits and stitches seamlessly)

#### Advanced
- **seed**: 0 for random, or fixed number for reproducibility
- **keep_model_in_vram**: Keep loaded for speed or unload to free ~8-11GB VRAM

#### Optional Input
- **prompt_audio**: Reference voice audio (3-30s recommended, 0.5-30s supported)

### Example Settings

**High Quality, Long Content**:
```
temperature: 0.7
do_sample: True
max_new_tokens: 8192
longform_chunking: True
torch_dtype: bfloat16
quantization: none
```

**Fast, Lower VRAM**:
```
temperature: 0.7
do_sample: True
max_new_tokens: 4096
longform_chunking: False
torch_dtype: float16
quantization: int8
```

**Consistent Results**:
```
temperature: 0.5
do_sample: True
seed: 42
max_new_tokens: 4096
```

---

## 🎭 Edit Node

### Overview

Edit existing audio with emotion, style, speed, paralinguistic effects, or denoising while preserving voice identity.

**Edit Types**:
- **Emotion**: happy, sad, angry, excited, calm, fearful, surprised, disgusted
- **Style**: whisper, gentle, serious, casual, formal, friendly
- **Speed**: faster (1.2x), slower (0.8x), more faster (1.5x), more slower (0.6x)
- **Paralinguistic**: [Laughter], [Breathing], [Sigh], [Gasp], [Cough]
- **Denoising**: denoise (remove background noise), vad (remove silence)

### Parameters

#### Text & Audio Inputs
- **audio_text**: Exact transcript of input audio
- **input_audio**: Audio to edit (0.5-30 seconds)

#### Model Configuration
Same as Clone Node (model_path, device, torch_dtype, quantization, attention_mechanism)

#### Edit Configuration
- **edit_type**: Select category (emotion/style/speed/paralinguistic/denoising)
- **emotion**: Target emotion (only if edit_type=emotion)
- **style**: Speaking style (only if edit_type=style)
- **speed**: Speed adjustment (only if edit_type=speed)
- **paralinguistic**: Sound effect (only if edit_type=paralinguistic)
- **denoising**: Noise removal (only if edit_type=denoising)
- **paralinguistic_text**: Where to insert effect (leave empty to auto-append to end)
- **n_edit_iterations**: Edit strength (1=subtle, 2-3=moderate, 4-5=strong)

#### Generation Parameters
- **temperature**: Hardcoded to 0.7 (parameter has no effect)
- **do_sample**: Hardcoded to True (parameter has no effect)
- **max_new_tokens**: Hardcoded to 8192 (parameter has no effect)

#### Advanced
- **seed**: 0 for random, or fixed number for reproducibility
- **keep_model_in_vram**: Keep loaded or unload to free VRAM

### Example Settings

**Add Emotion (Angry)**:
```
edit_type: emotion
emotion: angry
n_edit_iterations: 2
```

**Change Speaking Style (Whisper)**:
```
edit_type: style
style: whisper
n_edit_iterations: 3
```

**Speed Adjustment**:
```
edit_type: speed
speed: faster
n_edit_iterations: 1
```

**Add Laughter Effect**:
```
edit_type: paralinguistic
paralinguistic: [Laughter]
paralinguistic_text: (empty for auto-append to end)
n_edit_iterations: 1
```

**Clean Audio**:
```
edit_type: denoising
denoising: denoise
n_edit_iterations: 1
```

---

## 🎨 Workflow Examples

### Basic Voice Cloning

1. Load reference audio (3-30s of clear voice sample)
2. Add **Step Audio Clone Node**
3. Connect reference audio to `prompt_audio`
4. Enter exact transcript in `prompt_text`
5. Enter new text in `target_text`
6. Set temperature to 0.7
7. Generate!

### Long-Form Content with Chunking

1. Add **Step Audio Clone Node**
2. Enable `longform_chunking: True`
3. Set `max_new_tokens: 8192`
4. Enter long text (>2000 words) in `target_text`
5. Node will auto-split at sentence boundaries, generate chunks, and stitch seamlessly

### Edit Audio Emotion

1. Load source audio to edit
2. Add **Step Audio Edit Node**
3. Connect audio to `input_audio`
4. Enter exact transcript in `audio_text`
5. Set `edit_type: emotion`
6. Choose emotion (e.g., `happy`)
7. Set `n_edit_iterations: 2` for moderate strength
8. Generate!

### Clone + Edit Pipeline

1. Use **Clone Node** to generate speech in target voice
2. Connect clone output to **Edit Node** input
3. Apply emotion/style edits to the cloned voice
4. Fine-tune with multiple iterations for stronger effects

---

## ⚙️ Model Configuration Guide

### Precision (torch_dtype)

- **bfloat16**: Best quality, stable training format, recommended for RTX 40xx+
- **float16**: Good quality, compatible with most modern GPUs
- **float32**: Maximum quality, requires 16GB+ VRAM, overkill for most use cases
- **auto**: Automatically selects best precision for your GPU

### Quantization

- **none**: Best quality, highest VRAM usage (~11-14GB)
- **int8**: Good quality, medium VRAM (~9-11GB), recommended for VRAM-constrained systems
- **int4**: Acceptable quality, low VRAM, noticeable quality loss
- **int4_awq**: Optimized int4 quantization, slightly better than standard int4

### Attention Mechanism

- **sdpa**: Scaled Dot Product Attention (default, fastest, good VRAM efficiency)
- **eager**: Slowest but most stable, use for debugging
- **flash_attn**: Fastest option, requires RTX 30xx+ with Flash Attention support
- **sage_attn**: Best VRAM efficiency, slightly slower than sdpa

---

## 🐛 Troubleshooting

### Garbled or Distorted Speech

**This is often caused by dependency version mismatches. Try these solutions:**

1. **Update transformers to the correct version:**
   ```bash
   pip install transformers>=4.38.0
   ```

2. **Ensure required audio dependencies are installed:**
   ```bash
   pip install librosa hyperpyyaml
   ```

3. **Verify all dependencies are up to date:**
   ```bash
   cd ComfyUI/custom_nodes/ComfyUI-Step_Audio_TTS
   pip install -r requirements.txt --upgrade
   ```

4. **Check for conflicting installations:**
   - Uninstall and reinstall the node if issues persist
   - Clear ComfyUI's cache and restart

### Out of Memory (CUDA OOM)

1. Enable quantization (`int8` or `int4`)
2. Lower `max_new_tokens` (try 4096 instead of 8192)
3. Disable `keep_model_in_vram` after each generation
4. Use `float16` instead of `bfloat16`
5. Close other GPU applications
6. Try `sage_attn` attention mechanism for better VRAM efficiency

### Poor Voice Quality

1. Ensure `prompt_text` **exactly** matches reference audio transcript
2. Use higher quality reference audio (clean, minimal background noise, 3-30s)
3. Increase `temperature` (0.7-0.8) for more natural variation
4. Disable quantization for best quality
5. Use `bfloat16` precision
6. Verify reference audio is clear and well-recorded

### Edit Not Strong Enough

1. Increase `n_edit_iterations` (try 3-4 for stronger effects)
2. Ensure `audio_text` matches input audio transcript exactly
3. Verify you selected correct edit category option (not "none")
4. Try combining multiple edit passes for cumulative effects

### Model Not Loading

1. Check model path: `ComfyUI/models/Step-Audio-EditX/`
2. Ensure **both** Step-Audio-EditX and Step-Audio-Tokenizer are downloaded
3. Verify folder structure matches the required layout
4. Ensure sufficient disk space (models are ~10GB)
5. Check ComfyUI console for detailed error messages
6. Try re-downloading models if files are corrupted

### Slow Generation Speed

1. Use `flash_attn` attention mechanism if your GPU supports it
2. Enable `keep_model_in_vram` to avoid reload overhead
3. Use quantization (int8) for faster inference
4. Ensure you're using CUDA (not CPU)
5. Close background applications using GPU resources

---

## 📚 Credits & License

**Model**: [Step-Audio-EditX](https://huggingface.co/stepfun-ai/Step-Audio-EditX) by StepFun AI  
**ComfyUI Integration**: This custom node implementation  
**License**: MIT

---

## 🤝 Contributing

Contributions welcome! Please open issues for bugs or feature requests on our [GitHub repository](https://github.com/dannyson-sk/ComfyUI-Step_Audio_TTS/issues).

---

## 📝 Changelog

### Recent Updates

- ✅ Migrated to `max_new_tokens` for consistency across nodes
- ✅ Added paralinguistic auto-fill (auto-appends effect to end when text empty)
- ✅ Comprehensive tooltips for all parameters
- ✅ Fixed progress bar support (clone mode only)
- ✅ Enhanced VRAM management and caching
- ✅ Improved error handling and debugging messages

---

## 🔗 Links

- [Step Audio EditX Model](https://huggingface.co/stepfun-ai/Step-Audio-EditX)
- [Step Audio Tokenizer](https://huggingface.co/stepfun-ai/Step-Audio-Tokenizer)
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- [Report Issues](https://github.com/dannyson-sk/ComfyUI-Step_Audio_TTS/issues)
- [ComfyUI Manager](https://github.com/ltdrdata/ComfyUI-Manager)

---

## 💡 Tips for Best Results

- **Reference Audio**: Use clean, clear recordings without background noise
- **Transcripts**: Always match transcripts exactly - punctuation and spacing matter
- **Temperature**: Start with 0.7 and adjust based on results
- **Iterations**: For edits, start with 2 iterations and increase if needed
- **VRAM**: Monitor usage and adjust quantization settings accordingly
- **Long Content**: Enable chunking for texts over 2000 words for best results
