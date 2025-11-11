"""
Unified model loading utility supporting ModelScope, HuggingFace and local path loading
"""
import os
import json
import logging
import threading
from typing import Optional, Dict, Any, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Optional BitsAndBytes support (for quantization)
try:
    from transformers import BitsAndBytesConfig
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BitsAndBytesConfig = None
    BITSANDBYTES_AVAILABLE = False

from funasr_detach import AutoModel

# Optional AWQ support
try:
    from awq import AutoAWQForCausalLM
    AWQ_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    AutoAWQForCausalLM = None
    AWQ_AVAILABLE = False
    # Silently continue without AWQ support

# Optional safetensors support (for FP8 loading)
try:
    from safetensors.torch import load_file as load_safetensors
    SAFETENSORS_AVAILABLE = True
except ImportError:
    load_safetensors = None
    SAFETENSORS_AVAILABLE = False

# Global cache for downloaded models to avoid repeated downloads
# Key: (model_path, source)
# Value: local_model_path
_model_download_cache = {}
_download_cache_lock = threading.Lock()


class ModelSource:
    """Model source enumeration"""
    MODELSCOPE = "modelscope"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"
    AUTO = "auto"  # Auto-detect


class UnifiedModelLoader:
    """Unified model loader"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def _cached_snapshot_download(self, model_path: str, source: str, **kwargs) -> str:
        """
        Cached version of snapshot_download to avoid repeated downloads

        Args:
            model_path: Model path or ID to download
            source: Model source ('modelscope' or 'huggingface')
            **kwargs: Additional arguments for snapshot_download

        Returns:
            Local path to downloaded model
        """
        cache_key = (model_path, source, str(sorted(kwargs.items())))

        # Check cache first
        with _download_cache_lock:
            if cache_key in _model_download_cache:
                cached_path = _model_download_cache[cache_key]
                self.logger.info(f"Using cached download for {model_path} from {source}: {cached_path}")
                return cached_path

        # Cache miss, need to download
        if source == ModelSource.MODELSCOPE:
            from modelscope.hub.snapshot_download import snapshot_download
            local_path = snapshot_download(model_path, **kwargs)
        elif source == ModelSource.HUGGINGFACE:
            from huggingface_hub import snapshot_download
            local_path = snapshot_download(model_path, **kwargs)
        else:
            raise ValueError(f"Unsupported source for cached download: {source}")

        # Cache the result
        with _download_cache_lock:
            _model_download_cache[cache_key] = local_path

        self.logger.info(f"Downloaded and cached {model_path} from {source}: {local_path}")
        return local_path

    def detect_model_source(self, model_path: str) -> str:
        """
        Automatically detect model source

        Args:
            model_path: Model path or ID

        Returns:
            Model source type
        """
        # Local path detection
        if os.path.exists(model_path) or os.path.isabs(model_path):
            return ModelSource.LOCAL

        # ModelScope format detection (usually includes username/model_name)
        if "/" in model_path and not model_path.startswith("http"):
            # If contains modelscope keyword or is known modelscope format
            if "modelscope" in model_path.lower() or self._is_modelscope_format(model_path):
                return ModelSource.MODELSCOPE
            else:
                # Default to HuggingFace
                return ModelSource.HUGGINGFACE

        return ModelSource.LOCAL

    def _is_modelscope_format(self, model_path: str) -> bool:
        """Detect if it's ModelScope format model ID"""
        # Can be judged according to known ModelScope model ID formats
        # For example: iic/speech_paraformer-large_asr_nat-zh-cantonese-en-16k-vocab8501-online
        modelscope_patterns = []
        return any(pattern in model_path for pattern in modelscope_patterns)

    def _load_fp8_model(self, model_path: str, model_class, config, device_map: str = "auto") -> Any:
        """
        Load FP8 e4m3fn quantized model from safetensors with proper dtype conversion.

        This handles models that were quantized offline to FP8 format where:
        - Certain layers (attention, mlp) are stored as FP8 (as uint8 that needs conversion)
        - Other layers (embeddings, norms) remain in FP16
        - Metadata in model.safetensors.index.json specifies which layers are FP8

        Args:
            model_path: Path to model directory containing safetensors files
            model_class: Model class to instantiate
            config: Model configuration
            device_map: Device mapping for model loading

        Returns:
            Loaded model with FP8 weights
        """
        if not SAFETENSORS_AVAILABLE:
            raise ImportError("FP8 quantization requires 'safetensors' library. Install with: pip install safetensors")

        # Check for FP8 support (requires PyTorch 2.1+)
        if not hasattr(torch, 'float8_e4m3fn'):
            raise RuntimeError("FP8 e4m3fn requires PyTorch 2.1 or later with FP8 support")

        # Use print() instead of logger.info() since logging may be suppressed
        print(f"[StepAudio] 🔧 Loading FP8 e4m3fn quantized model from: {model_path}")

        # Load the safetensors index to get metadata
        index_path = os.path.join(model_path, "model.safetensors.index.json")
        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"FP8 model index not found at {index_path}. "
                f"Ensure your FP8 quantized model has model.safetensors.index.json with fp8_layers metadata."
            )

        with open(index_path, "r") as f:
            index_data = json.load(f)

        # Get FP8 layer list from metadata
        if "metadata" not in index_data or "fp8_layers" not in index_data["metadata"]:
            raise ValueError(
                f"Invalid FP8 model: model.safetensors.index.json must contain metadata.fp8_layers. "
                f"Found keys: {list(index_data.get('metadata', {}).keys())}"
            )

        fp8_layers = set(index_data["metadata"]["fp8_layers"])
        print(f"[StepAudio] 🔧 Found {len(fp8_layers)} FP8 layers in model metadata")

        # Load all safetensors files
        weight_map = index_data.get("weight_map", {})
        safetensors_files = set(weight_map.values())

        # Merge state dicts from all files
        state_dict = {}
        for st_file in safetensors_files:
            st_path = os.path.join(model_path, st_file)
            if not os.path.exists(st_path):
                raise FileNotFoundError(f"Safetensors file not found: {st_path}")

            print(f"[StepAudio] Loading weights from: {st_file}")
            file_state_dict = load_safetensors(st_path)
            state_dict.update(file_state_dict)

        print(f"[StepAudio] 🔧 Converting FP8 layers from uint8 storage...")
        # Convert uint8 FP8 layers back to torch.float8_e4m3fn
        converted_count = 0
        for layer_name in fp8_layers:
            if layer_name in state_dict:
                uint8_tensor = state_dict[layer_name]
                # Convert uint8 storage back to FP8 dtype
                fp8_tensor = uint8_tensor.view(torch.float8_e4m3fn)
                state_dict[layer_name] = fp8_tensor
                converted_count += 1

        print(f"[StepAudio] 🔧 Converted {converted_count} layers from uint8 to FP8 e4m3fn")

        # Check actual dtypes in state dict
        fp8_count = sum(1 for t in state_dict.values() if t.dtype == torch.float8_e4m3fn)
        fp16_count = sum(1 for t in state_dict.values() if t.dtype in (torch.float16, torch.bfloat16))
        print(f"[StepAudio] State dict dtypes: {fp8_count} FP8 tensors, {fp16_count} FP16/BF16 tensors")

        # Ensure config has all required attributes for model initialization
        # Add default values for missing attributes that are needed during init
        if not hasattr(config, 'initializer_range'):
            config.initializer_range = 0.02  # Standard default for transformer models
            print(f"[StepAudio] 🔧 Added missing initializer_range to config")

        # Create model with config using from_config() for AutoModelForCausalLM
        print(f"[StepAudio] 🔧 Initializing model architecture (this may take a while)...")
        import time
        start_time = time.time()
        model = model_class.from_config(config, trust_remote_code=True)
        init_time = time.time() - start_time
        print(f"[StepAudio] Model initialization took {init_time:.1f}s")

        # Load state dict with assign=True to preserve FP8 dtypes (PyTorch 2.0+)
        print(f"[StepAudio] 🔧 Loading FP8 state dict into model (preserving FP8 dtypes)...")
        start_time = time.time()
        try:
            # Try with assign=True first (PyTorch 2.0+) - preserves FP8 without conversion
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False, assign=True)
            print(f"[StepAudio] ✅ Used assign=True to preserve FP8 dtypes")
        except TypeError:
            # Fallback for older PyTorch versions without assign parameter
            print(f"[StepAudio] ⚠️  PyTorch version too old, assign=True not available")
            print(f"[StepAudio] ⚠️  FP8 weights will be converted to BF16 (no VRAM savings)")
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        load_time = time.time() - start_time
        print(f"[StepAudio] State dict loading took {load_time:.1f}s")

        # Check actual dtypes in loaded model
        model_dtypes = {}
        for name, param in model.named_parameters():
            dtype_name = str(param.dtype).replace('torch.', '')
            model_dtypes[dtype_name] = model_dtypes.get(dtype_name, 0) + 1
        print(f"[StepAudio] ⚠️  Model parameter dtypes after loading: {model_dtypes}")
        print(f"[StepAudio] ⚠️  WARNING: If no FP8 dtypes above, weights were converted to FP16/BF16!")

        if missing_keys:
            print(f"[StepAudio] ⚠️  Missing keys when loading FP8 model: {len(missing_keys)} keys")
        if unexpected_keys:
            print(f"[StepAudio] ⚠️  Unexpected keys when loading FP8 model: {len(unexpected_keys)} keys")

        # Move to device if specified
        if device_map != "auto":
            print(f"[StepAudio] Moving model to {device_map}...")
            model = model.to(device_map)

        # Register FP8 auto-casting hooks for inference
        print(f"[StepAudio] 🔧 Registering FP8→FP16 auto-cast hooks for inference...")
        self._register_fp8_hooks(model)

        print(f"[StepAudio] ✅ Successfully loaded FP8 e4m3fn model")
        return model

    def _register_fp8_hooks(self, model) -> int:
        """
        Wrap Linear layers to support FP8 storage with FP16 computation.

        This replaces the standard forward() method to:
        1. Store weights permanently in FP8 (saves VRAM)
        2. Cast FP8→FP16 temporarily for each matmul
        3. Keep FP8 storage unchanged

        Similar to how bitsandbytes handles INT4/INT8 quantization.

        Args:
            model: Model with FP8 weights

        Returns:
            Number of FP8 layers wrapped
        """
        import torch.nn as nn
        import torch.nn.functional as F

        wrapped_layers = 0

        print(f"[StepAudio] 🔧 Wrapping Linear layers for FP8 storage...")

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Check if weight is FP8
                if hasattr(module, 'weight') and module.weight.dtype == torch.float8_e4m3fn:
                    # Store FP8 weight in separate buffer
                    module.register_buffer('_weight_fp8', module.weight.data.clone())

                    # Store FP8 bias if present
                    if hasattr(module, 'bias') and module.bias is not None:
                        if module.bias.dtype == torch.float8_e4m3fn:
                            module.register_buffer('_bias_fp8', module.bias.data.clone())
                        else:
                            # Non-FP8 bias - keep as is
                            module.register_buffer('_bias_fp16', module.bias.data.clone())
                    else:
                        module._bias_fp8 = None
                        module._bias_fp16 = None

                    # Delete original parameters to save VRAM
                    del module.weight
                    if module.bias is not None:
                        del module.bias

                    # Replace forward method with FP8-aware version
                    original_forward = module.forward

                    def make_fp8_forward(mod):
                        """Create FP8-aware forward function for this module"""
                        def fp8_forward(input):
                            # Cast FP8 weights to FP16 (creates temporary copy, doesn't modify buffer)
                            weight_fp16 = mod._weight_fp8.to(torch.float16)

                            # Get bias
                            if hasattr(mod, '_bias_fp8') and mod._bias_fp8 is not None:
                                bias_fp16 = mod._bias_fp8.to(torch.float16)
                            elif hasattr(mod, '_bias_fp16') and mod._bias_fp16 is not None:
                                bias_fp16 = mod._bias_fp16
                            else:
                                bias_fp16 = None

                            # Perform linear operation with FP16 weights
                            return F.linear(input, weight_fp16, bias_fp16)

                        return fp8_forward

                    module.forward = make_fp8_forward(module)
                    module._is_fp8_wrapped = True
                    wrapped_layers += 1

        print(f"[StepAudio] ✅ Wrapped {wrapped_layers} Linear layers with FP8 storage")
        print(f"[StepAudio] 💡 Weights: Stored in FP8, cast to FP16 on-the-fly during forward")
        print(f"[StepAudio] 📊 VRAM savings: ~50% vs pure FP16 (~4-5GB saved)")

        return wrapped_layers

    def _prepare_quantization_config(self, quantization_config: Optional[str], torch_dtype: Optional[torch.dtype] = None) -> Tuple[Dict[str, Any], bool]:
        """
        Prepare quantization configuration for model loading

        Args:
            quantization_config: Quantization type ('int4', 'int8', 'int4_offline_awq', 'fp8_e4m3fn', or None)
            torch_dtype: PyTorch data type for compute operations

        Returns:
            Tuple of (quantization parameters dict, should_set_torch_dtype)
        """
        if not quantization_config:
            return {}, True

        quantization_config = quantization_config.lower()

        if quantization_config == "fp8_e4m3fn":
            # For pre-quantized FP8 models, no additional quantization needed at load time
            # The custom loader will handle FP8 conversion from stored uint8
            self.logger.info("🔧 Loading pre-quantized FP8 e4m3fn model (offline)")
            return {"_use_fp8_loader": True}, True  # Flag to use custom FP8 loader
        elif quantization_config == "int4_offline_awq":
            # For pre-quantized AWQ models, no additional quantization needed
            self.logger.info("🔧 Loading pre-quantized AWQ 4-bit model (offline)")
            return {}, True  # Load pre-quantized model normally, allow torch_dtype setting

        elif quantization_config == "int8":
            if not BITSANDBYTES_AVAILABLE:
                raise ImportError("INT8 quantization requested but 'bitsandbytes' is not installed. Please install with: pip install bitsandbytes")

            # Use user-specified torch_dtype for compute, default to bfloat16
            compute_dtype = torch_dtype if torch_dtype is not None else torch.bfloat16
            self.logger.info(f"🔧 INT8 quantization: using {compute_dtype} for compute operations")

            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=compute_dtype,
            )
            return {
                "quantization_config": bnb_config
            }, False  # INT8 quantization handles data types automatically, don't set torch_dtype
        elif quantization_config == "int4":
            if not BITSANDBYTES_AVAILABLE:
                raise ImportError("INT4 quantization requested but 'bitsandbytes' is not installed. Please install with: pip install bitsandbytes")

            # Use user-specified torch_dtype for compute, default to bfloat16
            compute_dtype = torch_dtype if torch_dtype is not None else torch.bfloat16
            self.logger.info(f"🔧 INT4 quantization: using {compute_dtype} for compute operations")

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )
            return {
                "quantization_config": bnb_config
            }, False  # INT4 quantization handles torch_dtype internally, don't set it again
        else:
            raise ValueError(f"Unsupported quantization config: {quantization_config}. Supported: 'int4', 'int8', 'int4_awq', 'fp8_e4m3fn'")

    def load_transformers_model(
        self,
        model_path: str,
        source: str = ModelSource.AUTO,
        quantization_config: Optional[str] = None,
        **kwargs
    ) -> Tuple:
        """
        Load Transformers model (for StepAudioTTS)

        Args:
            model_path: Model path or ID
            source: Model source, auto means auto-detect
            quantization_config: Quantization configuration ('int4', 'int8', 'int4_awq', 'fp8_e4m3fn', or None for no quantization)
            **kwargs: Other parameters (torch_dtype, device_map, etc.)

        Returns:
            (model, tokenizer, resolved_path) tuple
        """
        if source == ModelSource.AUTO:
            source = self.detect_model_source(model_path)

        self.logger.info(f"Loading Transformers model from {source}: {model_path}")
        if quantization_config:
            self.logger.info(f"🔧 {quantization_config.upper()} quantization enabled")

        # Prepare quantization configuration
        quantization_kwargs, should_set_torch_dtype = self._prepare_quantization_config(quantization_config, kwargs.get("torch_dtype"))

        try:
            if source == ModelSource.LOCAL:
                # Check if using FP8 custom loader
                if quantization_kwargs.get("_use_fp8_loader"):
                    # Load config first for FP8 model
                    from transformers import AutoConfig
                    config = AutoConfig.from_pretrained(
                        model_path,
                        trust_remote_code=True,
                        local_files_only=True
                    )

                    # Use custom FP8 loader
                    model = self._load_fp8_model(
                        model_path=model_path,
                        model_class=AutoModelForCausalLM,
                        config=config,
                        device_map=kwargs.get("device_map", "auto")
                    )

                    tokenizer = AutoTokenizer.from_pretrained(
                        model_path,
                        trust_remote_code=True,
                        local_files_only=True
                    )
                else:
                    # Standard loading path for non-FP8
                    load_kwargs = {
                        "device_map": kwargs.get("device_map", "auto"),
                        "trust_remote_code": True,
                        "local_files_only": True
                    }

                    # Add quantization configuration if specified
                    load_kwargs.update(quantization_kwargs)

                    # Add torch_dtype based on quantization requirements
                    if should_set_torch_dtype and kwargs.get("torch_dtype") is not None:
                        load_kwargs["torch_dtype"] = kwargs.get("torch_dtype")

                    # Check if using AWQ quantization
                    if quantization_config and quantization_config.lower() == "int4_offline_awq":
                        # Use AWQ loading for pre-quantized AWQ models
                        if not AWQ_AVAILABLE:
                            raise ImportError("AWQ quantization requested but 'autoawq' is not installed. Please install with: pip install autoawq")

                        awq_model_path = os.path.join(model_path, "awq_quantized")
                        if not os.path.exists(awq_model_path):
                            raise FileNotFoundError(f"AWQ quantized model not found at {awq_model_path}. Please run quantize_model_offline.py first.")

                        self.logger.info(f"🔧 Loading AWQ quantized model from: {awq_model_path}")
                        model = AutoAWQForCausalLM.from_quantized(
                            awq_model_path,
                            device_map=kwargs.get("device_map", "auto"),
                            trust_remote_code=True
                        )
                    else:
                        # Standard loading
                        model = AutoModelForCausalLM.from_pretrained(
                            model_path,
                            **load_kwargs
                        )

                    tokenizer = AutoTokenizer.from_pretrained(
                        model_path,
                        trust_remote_code=True,
                        local_files_only=True
                    )

            elif source == ModelSource.MODELSCOPE:
                # Load from ModelScope
                from modelscope import AutoModelForCausalLM as MSAutoModelForCausalLM
                from modelscope import AutoTokenizer as MSAutoTokenizer
                model_path = self._cached_snapshot_download(model_path, ModelSource.MODELSCOPE)

                load_kwargs = {
                    "device_map": kwargs.get("device_map", "auto"),
                    "trust_remote_code": True,
                    "local_files_only": True
                }

                # Add quantization configuration if specified
                load_kwargs.update(quantization_kwargs)

                # Add torch_dtype based on quantization requirements
                if should_set_torch_dtype and kwargs.get("torch_dtype") is not None:
                    load_kwargs["torch_dtype"] = kwargs.get("torch_dtype")

                # Check if using AWQ quantization
                if quantization_config and quantization_config.lower() == "int4_offline_awq":
                    # Use AWQ loading for pre-quantized AWQ models
                    if not AWQ_AVAILABLE:
                        raise ImportError("AWQ quantization requested but 'autoawq' is not installed. Please install with: pip install autoawq")

                    awq_model_path = os.path.join(model_path, "awq_quantized")
                    if not os.path.exists(awq_model_path):
                        raise FileNotFoundError(f"AWQ quantized model not found at {awq_model_path}. Please run quantize_model_offline.py first.")

                    self.logger.info(f"🔧 Loading AWQ quantized model from: {awq_model_path}")
                    model = AutoAWQForCausalLM.from_quantized(
                        awq_model_path,
                        device_map=kwargs.get("device_map", "auto"),
                        trust_remote_code=True
                    )
                else:
                    # Standard loading
                    model = MSAutoModelForCausalLM.from_pretrained(
                        model_path,
                        **load_kwargs
                    )
                tokenizer = MSAutoTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    local_files_only=True
                )

            elif source == ModelSource.HUGGINGFACE:
                model_path = self._cached_snapshot_download(model_path, ModelSource.HUGGINGFACE)

                # Load from HuggingFace
                load_kwargs = {
                    "device_map": kwargs.get("device_map", "auto"),
                    "trust_remote_code": True,
                    "local_files_only": True
                }

                # Add quantization configuration if specified
                load_kwargs.update(quantization_kwargs)

                # Add torch_dtype based on quantization requirements
                if should_set_torch_dtype and kwargs.get("torch_dtype") is not None:
                    load_kwargs["torch_dtype"] = kwargs.get("torch_dtype")

                # Check if using AWQ quantization
                if quantization_config and quantization_config.lower() == "int4_offline_awq":
                    # Use AWQ loading for pre-quantized AWQ models
                    if not AWQ_AVAILABLE:
                        raise ImportError("AWQ quantization requested but 'autoawq' is not installed. Please install with: pip install autoawq")

                    awq_model_path = os.path.join(model_path, "awq_quantized")
                    if not os.path.exists(awq_model_path):
                        raise FileNotFoundError(f"AWQ quantized model not found at {awq_model_path}. Please run quantize_model_offline.py first.")

                    self.logger.info(f"🔧 Loading AWQ quantized model from: {awq_model_path}")
                    model = AutoAWQForCausalLM.from_quantized(
                        awq_model_path,
                        device_map=kwargs.get("device_map", "auto"),
                        trust_remote_code=True
                    )
                else:
                    # Standard loading
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        **load_kwargs
                    )
                tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    local_files_only=True
                )

            else:
                raise ValueError(f"Unsupported model source: {source}")

            self.logger.info(f"Successfully loaded model from {source}")
            return model, tokenizer, model_path

        except Exception as e:
            self.logger.error(f"Failed to load model from {source}: {e}")
            raise

    def load_funasr_model(
        self,
        repo_path: str,
        model_path: str,
        source: str = ModelSource.AUTO,
        **kwargs
    ) -> AutoModel:
        """
        Load FunASR model (for StepAudioTokenizer)

        Args:
            model_path: Model path or ID
            source: Model source, auto means auto-detect
            **kwargs: Other parameters

        Returns:
            FunASR AutoModel instance
        """
        if source == ModelSource.AUTO:
            source = self.detect_model_source(model_path)
            
        self.logger.info(f"Loading FunASR model from {source}: {model_path}")

        try:
            # Extract model_revision to avoid duplicate passing
            model_revision = kwargs.pop("model_revision", "main")

            # Map ModelSource to model_hub parameter
            if source == ModelSource.LOCAL:
                model_hub = "local"
            elif source == ModelSource.MODELSCOPE:
                model_hub = "ms"
            elif source == ModelSource.HUGGINGFACE:
                model_hub = "hf"
            else:
                raise ValueError(f"Unsupported model source: {source}")

            # Use unified download_model for all cases
            model = AutoModel(
                repo_path=repo_path,
                model=model_path,
                model_hub=model_hub,
                model_revision=model_revision,
                **kwargs
            )

            self.logger.info(f"Successfully loaded FunASR model from {source}")
            return model

        except Exception as e:
            self.logger.error(f"Failed to load FunASR model from {source}: {e}")
            raise

    def resolve_model_path(
        self,
        base_path: str,
        model_name: str,
        source: str = ModelSource.AUTO
    ) -> str:
        """
        Resolve model path

        Args:
            base_path: Base path
            model_name: Model name
            source: Model source

        Returns:
            Resolved model path
        """
        if source == ModelSource.AUTO:
            # First check local path
            local_path = os.path.join(base_path, model_name)
            if os.path.exists(local_path):
                return local_path

            # If local doesn't exist, return model name for online download
            return model_name

        elif source == ModelSource.LOCAL:
            return os.path.join(base_path, model_name)

        else:
            # For online sources, directly return model name/ID
            return model_name


# Global instance
model_loader = UnifiedModelLoader()