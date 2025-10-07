"""
High-level encoder API for creating .wcodec files
"""

import numpy as np
from pathlib import Path
from typing import Union, Dict, Optional
from .bindings import Encoder

try:
    import safetensors
    from safetensors import safe_open
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False


def encode_checkpoint(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    tile_size: int = 16,
    model_name: Optional[str] = None,
    quantization_type: str = "int8",
    verbose: bool = True
) -> Dict:
    """
    Encode a safetensors checkpoint to .wcodec format
    
    Args:
        input_path: Path to input .safetensors file
        output_path: Path to output .wcodec file
        tile_size: Tile size for encoding (default: 16)
        model_name: Model name for metadata
        quantization_type: Quantization type (default: "int8")
        verbose: Print progress
    
    Returns:
        dict: Encoding statistics
            - num_layers: Number of layers encoded
            - total_uncompressed: Total uncompressed size (bytes)
            - total_compressed: Total compressed size (bytes)
            - compression_ratio: Overall compression ratio
            - per_layer_stats: List of per-layer statistics
    
    Example:
        >>> stats = encode_checkpoint(
        ...     "model.safetensors",
        ...     "model.wcodec",
        ...     tile_size=16
        ... )
        >>> print(f"Compression: {stats['compression_ratio']:.2f}x")
    """
    if not SAFETENSORS_AVAILABLE:
        raise ImportError("safetensors library required. Install with: pip install safetensors")
    
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    if model_name is None:
        model_name = input_path.stem
    
    if verbose:
        print(f"Encoding {input_path.name} → {output_path.name}")
        print(f"Tile size: {tile_size}×{tile_size}")
    
    # Load safetensors
    tensors = {}
    with safe_open(input_path, framework="np") as f:
        tensor_names = f.keys()
        for name in tensor_names:
            tensors[name] = f.get_tensor(name)
    
    if verbose:
        print(f"Loaded {len(tensors)} tensors")
    
    # Create encoder
    encoder = Encoder(tile_size=tile_size)
    
    # Encode each layer
    per_layer_stats = []
    total_uncompressed = 0
    total_compressed = 0
    
    # For now, write to temporary format (full container format integration pending)
    # In production, this would use ContainerWriter C++ class
    
    for i, (name, tensor) in enumerate(tensors.items()):
        # Convert to int8 if needed
        if tensor.dtype != np.int8:
            if verbose:
                print(f"  [{i+1}/{len(tensors)}] Skipping {name} (dtype={tensor.dtype}, only int8 supported)")
            continue
        
        # Only handle 2D tensors for now
        if tensor.ndim != 2:
            if verbose:
                print(f"  [{i+1}/{len(tensors)}] Skipping {name} (shape={tensor.shape}, only 2D supported)")
            continue
        
        rows, cols = tensor.shape
        
        if verbose:
            print(f"  [{i+1}/{len(tensors)}] Encoding {name} [{rows}×{cols}]...", end=" ")
        
        # Encode
        compressed, stats = encoder.encode_layer(tensor)
        
        # Track stats
        layer_stat = {
            'name': name,
            'shape': (rows, cols),
            'uncompressed_bytes': stats['original_bytes'],
            'compressed_bytes': stats['compressed_bytes'],
            'compression_ratio': stats['compression_ratio'],
            'encode_time_ms': stats['encode_time_ms']
        }
        per_layer_stats.append(layer_stat)
        
        total_uncompressed += stats['original_bytes']
        total_compressed += stats['compressed_bytes']
        
        if verbose:
            print(f"{stats['compression_ratio']:.2f}x ({stats['encode_time_ms']:.1f}ms)")
    
    # Calculate overall stats
    overall_ratio = total_uncompressed / max(1, total_compressed)
    
    # TODO: Write actual .wcodec file using ContainerWriter
    # For now, just return stats
    
    if verbose:
        print(f"\nTotal: {total_uncompressed / (1024**2):.1f} MB → "
              f"{total_compressed / (1024**2):.1f} MB ({overall_ratio:.2f}x)")
        print(f"Output: {output_path} (placeholder - container format pending)")
    
    return {
        'num_layers': len(per_layer_stats),
        'total_uncompressed': total_uncompressed,
        'total_compressed': total_compressed,
        'compression_ratio': overall_ratio,
        'per_layer_stats': per_layer_stats,
        'model_name': model_name,
        'quantization_type': quantization_type,
        'tile_size': tile_size
    }


def encode_layer_standalone(
    data: np.ndarray,
    output_path: Union[str, Path],
    tile_size: int = 16,
    layer_name: str = "layer"
) -> Dict:
    """
    Encode a single layer to a file
    
    Args:
        data: 2D int8 numpy array
        output_path: Output file path
        tile_size: Tile size
        layer_name: Layer name for metadata
    
    Returns:
        dict: Encoding statistics
    """
    if data.dtype != np.int8:
        raise ValueError(f"Expected int8 data, got {data.dtype}")
    
    if data.ndim != 2:
        raise ValueError(f"Expected 2D array, got {data.ndim}D")
    
    encoder = Encoder(tile_size=tile_size)
    compressed, stats = encoder.encode_layer(data)
    
    # Write to file
    output_path = Path(output_path)
    with open(output_path, 'wb') as f:
        f.write(compressed)
    
    return stats


__all__ = [
    'encode_checkpoint',
    'encode_layer_standalone',
]

