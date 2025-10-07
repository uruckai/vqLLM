#!/usr/bin/env python3
"""
CLI tool to encode safetensors checkpoint to .wcodec format
"""

import sys
import argparse
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from wcodec.encoder_api import encode_checkpoint


def main():
    parser = argparse.ArgumentParser(
        description="Encode safetensors checkpoint to .wcodec format"
    )
    parser.add_argument(
        "input",
        type=str,
        help="Input .safetensors file"
    )
    parser.add_argument(
        "output",
        type=str,
        help="Output .wcodec file"
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=16,
        help="Tile size for encoding (default: 16)"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Model name for metadata (default: input filename)"
    )
    parser.add_argument(
        "--quant-type",
        type=str,
        default="int8",
        choices=["int8", "int4", "fp8"],
        help="Quantization type (default: int8)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )
    
    args = parser.parse_args()
    
    try:
        stats = encode_checkpoint(
            input_path=args.input,
            output_path=args.output,
            tile_size=args.tile_size,
            model_name=args.model_name,
            quantization_type=args.quant_type,
            verbose=not args.quiet
        )
        
        if not args.quiet:
            print(f"\n✓ Encoding complete!")
            print(f"  Layers: {stats['num_layers']}")
            print(f"  Compression: {stats['compression_ratio']:.2f}x")
            print(f"  Original: {stats['total_uncompressed'] / (1024**2):.1f} MB")
            print(f"  Compressed: {stats['total_compressed'] / (1024**2):.1f} MB")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

