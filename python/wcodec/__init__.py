"""
Weight Codec (WCodec) - Storage codec for LLM weights

A codec inspired by video compression techniques (AV1/VP9) that achieves
30-60% smaller checkpoint files with bit-exact reconstruction.
"""

__version__ = "0.1.0"

# Low-level bindings (Week 2+3+6)
try:
    from .bindings import Encoder, Decoder, GPUDecoder, is_gpu_available
    _bindings_available = True
except ImportError:
    _bindings_available = False
    Encoder = None
    Decoder = None
    GPUDecoder = None
    is_gpu_available = lambda: False

# High-level APIs (Week 5)
try:
    from .encoder_api import encode_checkpoint as _encode_checkpoint
    from .decoder_api import decode_checkpoint as _decode_checkpoint, load_to_torch
    _high_level_available = True
except ImportError:
    _high_level_available = False
    _encode_checkpoint = None
    _decode_checkpoint = None
    load_to_torch = None

# For now, provide stubs for documentation
def encode_checkpoint(*args, **kwargs):
    """
    Encode a quantized checkpoint to .wcodec format.
    
    Args:
        input_path (str): Path to input checkpoint (.safetensors)
        output_path (str): Path to output .wcodec file
        tile_size (int): Tile size (default: 16)
        predictor_modes (list): Predictor modes to use
        transforms (list): Transform types to use
        threads (int): Number of threads for encoding
    
    Returns:
        dict: Encoding statistics
    
    Note:
        Week 5: Partial implementation (container format pending)
    """
    if _high_level_available and _encode_checkpoint:
        return _encode_checkpoint(*args, **kwargs)
    raise NotImplementedError("Encoder implementation in progress (Week 5)")


def decode_checkpoint(*args, **kwargs):
    """
    Decode a .wcodec file to safetensors format.
    
    Args:
        input_path (str): Path to .wcodec file
        output_path (str, optional): Path to output .safetensors file
        device (str): Target device ("cuda" or "cpu")
        fallback (str): Fallback device if primary unavailable
    
    Returns:
        dict or None: State dict if output_path is None, else None
    
    Note:
        Week 5: Partial implementation (container format pending)
    """
    if _high_level_available and _decode_checkpoint:
        return _decode_checkpoint(*args, **kwargs)
    raise NotImplementedError("Decoder implementation in progress (Week 5)")


def load_model(*args, **kwargs):
    """
    Load a model directly from .wcodec file.
    
    Args:
        wcodec_path (str): Path to .wcodec file
        device (str): Target device ("cuda" or "cpu")
    
    Returns:
        torch.nn.Module: Loaded model
    
    Note:
        Implementation coming in Week 5
    """
    raise NotImplementedError("load_model will be implemented in Week 5")


def is_cuda_available():
    """Check if CUDA decoder is available."""
    return False  # Will be implemented in Week 4


def set_decode_params(*args, **kwargs):
    """
    Set GPU decode parameters.
    
    Args:
        tiles_per_block (int): Tiles per CUDA block
        streams (int): Number of CUDA streams
        use_tensor_cores (bool): Use tensor cores for inverse transforms
    
    Note:
        Implementation coming in Week 4
    """
    raise NotImplementedError("GPU decode parameters will be configurable in Week 4")


__all__ = [
    "encode_checkpoint",
    "decode_checkpoint",
    "load_model",
    "load_to_torch",
    "is_cuda_available",
    "is_gpu_available",
    "set_decode_params",
    "Encoder",
    "Decoder",
    "GPUDecoder",
]

