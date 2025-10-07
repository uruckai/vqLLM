"""
High-level decoder API for loading .wcodec files
"""

import numpy as np
from pathlib import Path
from typing import Union, Dict, Optional
from .bindings import Decoder

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def decode_checkpoint(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    device: str = "cpu",
    use_gpu: bool = False,
    verbose: bool = True
) -> Optional[Dict[str, np.ndarray]]:
    """
    Decode a .wcodec file to numpy arrays or safetensors
    
    Args:
        input_path: Path to .wcodec file
        output_path: Optional output .safetensors path (if None, returns state dict)
        device: Target device for tensors ("cpu" or "cuda")
        use_gpu: Use GPU decoder if available
        verbose: Print progress
    
    Returns:
        dict: State dict with numpy arrays (if output_path is None)
        None: If output_path is provided (writes to file)
    
    Example:
        >>> # Load to memory
        >>> state_dict = decode_checkpoint("model.wcodec")
        >>> 
        >>> # Write to safetensors
        >>> decode_checkpoint("model.wcodec", "model.safetensors")
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    if verbose:
        print(f"Decoding {input_path.name}")
        if use_gpu:
            print("GPU decode: requested (implementation pending)")
        else:
            print("CPU decode: active")
    
    # TODO: Use ContainerReader to parse .wcodec file
    # For now, placeholder implementation
    
    if verbose:
        print("Note: Full container format pending - using placeholder")
    
    state_dict = {}
    
    # Placeholder: Would iterate through layers and decode each
    # decoder = Decoder(tile_size=16)
    # for layer_info in container.layers:
    #     compressed = container.read_layer_data(layer_info.name)
    #     decoded, stats = decoder.decode_layer(compressed, layer_info.rows, layer_info.cols)
    #     state_dict[layer_info.name] = decoded
    
    if output_path is not None:
        # Write to safetensors
        if verbose:
            print(f"Writing to {output_path}")
        # TODO: Implement safetensors writing
        raise NotImplementedError("Container format not yet complete")
    else:
        return state_dict


def decode_layer_standalone(
    input_path: Union[str, Path],
    rows: int,
    cols: int,
    tile_size: int = 16,
    use_gpu: bool = False
) -> np.ndarray:
    """
    Decode a single layer from a standalone file
    
    Args:
        input_path: Path to compressed layer file
        rows: Number of rows
        cols: Number of columns
        tile_size: Tile size used during encoding
        use_gpu: Use GPU decoder if available
    
    Returns:
        np.ndarray: Decoded int8 array
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Read compressed data
    with open(input_path, 'rb') as f:
        compressed = f.read()
    
    # Decode
    decoder = Decoder(tile_size=tile_size)
    decoded, stats = decoder.decode_layer(compressed, rows, cols)
    
    return decoded


def load_to_torch(
    input_path: Union[str, Path],
    device: str = "cpu",
    use_gpu_decode: bool = True
) -> Dict[str, "torch.Tensor"]:
    """
    Load .wcodec checkpoint directly to PyTorch tensors
    
    Args:
        input_path: Path to .wcodec file
        device: Target torch device
        use_gpu_decode: Use GPU decoder if available
    
    Returns:
        dict: State dict with torch tensors
    
    Example:
        >>> state_dict = load_to_torch("model.wcodec", device="cuda")
        >>> model.load_state_dict(state_dict)
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required. Install with: pip install torch")
    
    # Decode to numpy
    numpy_dict = decode_checkpoint(input_path, use_gpu=use_gpu_decode, verbose=False)
    
    # Convert to torch tensors
    torch_dict = {}
    for name, array in numpy_dict.items():
        tensor = torch.from_numpy(array).to(device)
        torch_dict[name] = tensor
    
    return torch_dict


__all__ = [
    'decode_checkpoint',
    'decode_layer_standalone',
    'load_to_torch',
]

