#!/usr/bin/env python3
"""
Checkpoint Verification Tool

Verifies bit-exact reconstruction after encode/decode cycle.

Usage:
    python verify_checkpoint.py --original model.safetensors --decoded model_decoded.safetensors
"""

import argparse
import sys
from pathlib import Path

import torch
import numpy as np
from safetensors.torch import load_file


def compare_tensors(original: dict, decoded: dict, tolerance: float = 0.0) -> dict:
    """
    Compare two state dicts for equality.
    
    Args:
        original: Original state dict
        decoded: Decoded state dict
        tolerance: Absolute tolerance for floating point comparison (0.0 for exact)
    
    Returns:
        Dictionary with comparison results
    """
    results = {
        "total_layers": len(original),
        "matching_layers": 0,
        "mismatched_layers": [],
        "missing_layers": [],
        "extra_layers": [],
        "max_abs_diff": 0.0,
        "mean_abs_diff": 0.0,
        "exact_match": True
    }
    
    # Check for missing/extra layers
    original_keys = set(original.keys())
    decoded_keys = set(decoded.keys())
    
    results["missing_layers"] = list(original_keys - decoded_keys)
    results["extra_layers"] = list(decoded_keys - original_keys)
    
    # Compare matching layers
    all_diffs = []
    
    for name in original_keys & decoded_keys:
        orig_tensor = original[name]
        dec_tensor = decoded[name]
        
        # Check shape
        if orig_tensor.shape != dec_tensor.shape:
            results["mismatched_layers"].append({
                "name": name,
                "reason": "shape_mismatch",
                "original_shape": list(orig_tensor.shape),
                "decoded_shape": list(dec_tensor.shape)
            })
            results["exact_match"] = False
            continue
        
        # Check dtype
        if orig_tensor.dtype != dec_tensor.dtype:
            results["mismatched_layers"].append({
                "name": name,
                "reason": "dtype_mismatch",
                "original_dtype": str(orig_tensor.dtype),
                "decoded_dtype": str(dec_tensor.dtype)
            })
            results["exact_match"] = False
            continue
        
        # Compare values
        if tolerance == 0.0:
            # Exact comparison
            if torch.equal(orig_tensor, dec_tensor):
                results["matching_layers"] += 1
            else:
                diff = (orig_tensor != dec_tensor).sum().item()
                max_diff = (orig_tensor - dec_tensor).abs().max().item() if orig_tensor.dtype.is_floating_point else diff
                
                results["mismatched_layers"].append({
                    "name": name,
                    "reason": "value_mismatch",
                    "num_different": diff,
                    "max_abs_diff": max_diff,
                    "total_elements": orig_tensor.numel()
                })
                results["exact_match"] = False
        else:
            # Tolerance-based comparison
            diff = (orig_tensor.float() - dec_tensor.float()).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            
            all_diffs.append(diff.flatten())
            
            if max_diff <= tolerance:
                results["matching_layers"] += 1
            else:
                results["mismatched_layers"].append({
                    "name": name,
                    "reason": "tolerance_exceeded",
                    "max_abs_diff": max_diff,
                    "mean_abs_diff": mean_diff,
                    "tolerance": tolerance
                })
                results["exact_match"] = False
    
    # Compute global statistics
    if all_diffs:
        all_diffs = torch.cat(all_diffs)
        results["max_abs_diff"] = all_diffs.max().item()
        results["mean_abs_diff"] = all_diffs.mean().item()
    
    return results


def print_results(results: dict, verbose: bool = False):
    """Print comparison results in human-readable format."""
    print(f"\n{'='*60}")
    print("CHECKPOINT VERIFICATION RESULTS")
    print(f"{'='*60}\n")
    
    # Summary
    print(f"Total layers: {results['total_layers']}")
    print(f"Matching layers: {results['matching_layers']}")
    print(f"Mismatched layers: {len(results['mismatched_layers'])}")
    print(f"Missing layers: {len(results['missing_layers'])}")
    print(f"Extra layers: {len(results['extra_layers'])}")
    print()
    
    if results["exact_match"]:
        print("✓ PASS: All layers match bit-exactly!")
    else:
        print("✗ FAIL: Layers do not match!")
        print(f"  Max absolute difference: {results['max_abs_diff']:.6e}")
        print(f"  Mean absolute difference: {results['mean_abs_diff']:.6e}")
    
    # Detailed mismatch info
    if results["mismatched_layers"] and verbose:
        print(f"\n{'='*60}")
        print("MISMATCHED LAYERS (detailed)")
        print(f"{'='*60}\n")
        for mismatch in results["mismatched_layers"]:
            print(f"Layer: {mismatch['name']}")
            print(f"  Reason: {mismatch['reason']}")
            for key, value in mismatch.items():
                if key not in ["name", "reason"]:
                    print(f"  {key}: {value}")
            print()
    elif results["mismatched_layers"]:
        print(f"\nMismatched layers (use --verbose for details):")
        for mismatch in results["mismatched_layers"][:10]:  # Show first 10
            print(f"  - {mismatch['name']}: {mismatch['reason']}")
        if len(results["mismatched_layers"]) > 10:
            print(f"  ... and {len(results['mismatched_layers']) - 10} more")
    
    # Missing layers
    if results["missing_layers"]:
        print(f"\nMissing layers in decoded checkpoint:")
        for name in results["missing_layers"][:10]:
            print(f"  - {name}")
        if len(results["missing_layers"]) > 10:
            print(f"  ... and {len(results['missing_layers']) - 10} more")
    
    # Extra layers
    if results["extra_layers"]:
        print(f"\nExtra layers in decoded checkpoint:")
        for name in results["extra_layers"][:10]:
            print(f"  - {name}")
        if len(results["extra_layers"]) > 10:
            print(f"  ... and {len(results['extra_layers']) - 10} more")
    
    print(f"\n{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Verify checkpoint reconstruction")
    parser.add_argument("--original", type=str, required=True,
                       help="Path to original checkpoint (.safetensors)")
    parser.add_argument("--decoded", type=str, required=True,
                       help="Path to decoded checkpoint (.safetensors)")
    parser.add_argument("--tolerance", type=float, default=0.0,
                       help="Absolute tolerance for floating point comparison (0.0 for exact)")
    parser.add_argument("--verbose", action="store_true",
                       help="Print detailed mismatch information")
    parser.add_argument("--output", type=str, default=None,
                       help="Save results to JSON file")
    
    args = parser.parse_args()
    
    # Load checkpoints
    print(f"Loading original checkpoint: {args.original}")
    original = load_file(args.original, device="cpu")
    
    print(f"Loading decoded checkpoint: {args.decoded}")
    decoded = load_file(args.decoded, device="cpu")
    
    # Compare
    print("Comparing tensors...")
    results = compare_tensors(original, decoded, tolerance=args.tolerance)
    
    # Print results
    print_results(results, verbose=args.verbose)
    
    # Save to JSON if requested
    if args.output:
        import json
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {output_path}")
    
    # Exit code
    sys.exit(0 if results["exact_match"] else 1)


if __name__ == "__main__":
    main()

