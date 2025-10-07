#!/usr/bin/env python3
"""
CLI tool to decode .wcodec file to safetensors format
"""

import sys
import argparse
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from wcodec.decoder_api import decode_checkpoint


def main():
    parser = argparse.ArgumentParser(
        description="Decode .wcodec file to safetensors format"
    )
    parser.add_argument(
        "input",
        type=str,
        help="Input .wcodec file"
    )
    parser.add_argument(
        "output",
        type=str,
        nargs="?",
        default=None,
        help="Output .safetensors file (optional, prints info if not provided)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Target device (default: cpu)"
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use GPU decoder if available"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )
    
    args = parser.parse_args()
    
    try:
        result = decode_checkpoint(
            input_path=args.input,
            output_path=args.output,
            device=args.device,
            use_gpu=args.use_gpu,
            verbose=not args.quiet
        )
        
        if args.output is None:
            # Print info mode
            if not args.quiet:
                print(f"\n✓ Decoding complete!")
                print(f"  Layers: {len(result)}")
                for name, tensor in result.items():
                    print(f"    {name}: {tensor.shape}")
        else:
            if not args.quiet:
                print(f"\n✓ Written to {args.output}")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

