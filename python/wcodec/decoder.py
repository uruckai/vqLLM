"""
Python wrapper for Weight Codec decoder
"""

import ctypes
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import torch


# Try to load the C++ library
def _load_library():
    """Load the wcodec shared library"""
    search_paths = [
        Path(__file__).parent.parent.parent / "build" / "libwcodec.so",  # Linux
        Path(__file__).parent.parent.parent / "build" / "libwcodec.dylib",  # macOS
        Path(__file__).parent.parent.parent / "build" / "Release" / "wcodec.dll",  # Windows
        Path(__file__).parent.parent.parent / "build" / "Debug" / "wcodec.dll",  # Windows Debug
    ]
    
    for path in search_paths:
        if path.exists():
            return ctypes.CDLL(str(path))
    
    return None


_lib = _load_library()


def decode_checkpoint(
    input_path: str,
    output_path: Optional[str] = None,
    device: str = "cuda",
    fallback: str = "cpu"
) -> Optional[Dict[str, torch.Tensor]]:
    """
    Decode a .wcodec file to safetensors format or state dict.
    
    Args:
        input_path: Path to .wcodec file
        output_path: Path to output .safetensors file (optional)
        device: Target device ("cuda" or "cpu")
        fallback: Fallback device if primary unavailable
    
    Returns:
        State dict if output_path is None, else None
    
    Note:
        C++ decoder is implemented but not yet exposed via Python bindings.
        Will be fully functional after adding pybind11 wrappers (Week 2 end).
    """
    if _lib is None:
        raise RuntimeError(
            "C++ library not found. Please build the project first:\n"
            "  mkdir build && cd build\n"
            "  cmake .. -DCMAKE_BUILD_TYPE=Release\n"
            "  make -j8"
        )
    
    # TODO: Add pybind11 bindings to expose decoder
    raise NotImplementedError(
        "Python bindings not yet complete. "
        "C++ decoder is implemented - need to add pybind11 wrapper."
    )


def decode_layer_numpy(
    data: bytes,
    rows: int,
    cols: int,
    tile_size: int = 16
) -> np.ndarray:
    """
    Decode compressed bytes to numpy array.
    
    Args:
        data: Compressed bytes
        rows: Expected number of rows
        cols: Expected number of columns
        tile_size: Tile size (must match encoder)
    
    Returns:
        np.ndarray: Decoded array (int8)
    
    Note:
        C++ implementation complete, Python bindings coming in Week 2 end.
    """
    if _lib is None:
        raise RuntimeError("C++ library not found. Please build the project first.")
    
    # TODO: Call C++ decoder via ctypes or pybind11
    raise NotImplementedError("Python bindings not yet complete.")


def load_model(
    wcodec_path: str,
    device: str = "cuda"
) -> torch.nn.Module:
    """
    Load a model directly from .wcodec file.
    
    Args:
        wcodec_path: Path to .wcodec file
        device: Target device ("cuda" or "cpu")
    
    Returns:
        torch.nn.Module: Loaded model
    
    Note:
        Full integration coming in Week 5.
    """
    raise NotImplementedError("load_model will be implemented in Week 5")

