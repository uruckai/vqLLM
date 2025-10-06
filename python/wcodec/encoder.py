"""
Python wrapper for Weight Codec encoder
"""

import ctypes
import numpy as np
from pathlib import Path
from typing import Dict, Any
import os


# Try to load the C++ library
def _load_library():
    """Load the wcodec shared library"""
    # Search paths
    search_paths = [
        Path(__file__).parent.parent.parent / "build" / "libwcodec.so",  # Linux
        Path(__file__).parent.parent.parent / "build" / "libwcodec.dylib",  # macOS
        Path(__file__).parent.parent.parent / "build" / "Release" / "wcodec.dll",  # Windows
        Path(__file__).parent.parent.parent / "build" / "Debug" / "wcodec.dll",  # Windows Debug
    ]
    
    for path in search_paths:
        if path.exists():
            return ctypes.CDLL(str(path))
    
    # Not found
    return None


_lib = _load_library()


def encode_checkpoint(
    input_path: str,
    output_path: str,
    tile_size: int = 16,
    predictor_modes: list = None,
    transforms: list = None,
    threads: int = None
) -> Dict[str, Any]:
    """
    Encode a quantized checkpoint to .wcodec format.
    
    Args:
        input_path: Path to input checkpoint (.safetensors)
        output_path: Path to output .wcodec file
        tile_size: Tile size (default: 16)
        predictor_modes: Predictor modes to use
        transforms: Transform types to use (Week 3)
        threads: Number of threads for encoding
    
    Returns:
        dict: Encoding statistics
    
    Note:
        C++ implementation complete but not yet exposed via Python bindings.
        Will be fully functional after adding pybind11 wrappers (Week 2 end).
    """
    if _lib is None:
        raise RuntimeError(
            "C++ library not found. Please build the project first:\n"
            "  mkdir build && cd build\n"
            "  cmake .. -DCMAKE_BUILD_TYPE=Release\n"
            "  make -j8"
        )
    
    # TODO: Add pybind11 bindings to expose encoder
    # For now, this is a placeholder
    raise NotImplementedError(
        "Python bindings not yet complete. "
        "C++ encoder is implemented - need to add pybind11 wrapper."
    )


def encode_layer_numpy(
    data: np.ndarray,
    tile_size: int = 16
) -> bytes:
    """
    Encode a single layer (numpy array) to compressed bytes.
    
    Args:
        data: 2D numpy array (int8)
        tile_size: Tile size
    
    Returns:
        bytes: Compressed data
    
    Note:
        C++ implementation complete, Python bindings coming in Week 2 end.
    """
    if _lib is None:
        raise RuntimeError("C++ library not found. Please build the project first.")
    
    if data.dtype != np.int8:
        raise ValueError(f"Expected int8 data, got {data.dtype}")
    
    if data.ndim != 2:
        raise ValueError(f"Expected 2D array, got {data.ndim}D")
    
    # TODO: Call C++ encoder via ctypes or pybind11
    raise NotImplementedError("Python bindings not yet complete.")

