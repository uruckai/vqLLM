#!/usr/bin/env python3
"""
Diagnose why output shows question marks instead of proper text
"""

import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# Test string with problematic output
baseline_text = "The capital of France is Paris.\n\n2. B. The capital"
compressed_155_text = "The capital of France is??????????"
compressed_20_text = "The capital of France is����������"

print("="*80)
print("OUTPUT ENCODING DIAGNOSIS")
print("="*80)
print()

# Check byte representations
print("1. Byte-level analysis:")
print(f"Baseline (first 30 chars): {baseline_text[:30]}")
print(f"  Bytes: {baseline_text[:30].encode('utf-8')}")
print()

print(f"155 layers (first 30): {compressed_155_text[:30]}")
print(f"  Bytes: {compressed_155_text[:30].encode('utf-8', errors='replace')}")
print(f"  Length: {len(compressed_155_text)}")
print()

print(f"20 layers (first 30): {compressed_20_text[:30]}")
print(f"  Bytes: {compressed_20_text[:30].encode('utf-8', errors='replace')}")
print(f"  Length: {len(compressed_20_text)}")
print()

# Check character codes
print("2. Character code analysis:")
print("Baseline 'P' (Paris):", ord('P'), "- ASCII")
print("155 layers '?' char:", ord('?') if '?' in compressed_155_text else "N/A")

# Check if it's a tokenization issue
print()
print("3. Token analysis:")
print("Both outputs start with 'The capital of France is' (✓ correct)")
print("Then diverge at character position ~25")
print()

# Hypothesis
print("="*80)
print("HYPOTHESIS")
print("="*80)
print()
print("The question marks and undefined symbols suggest:")
print("  1. Quantization is producing INVALID logits")
print("  2. Model is generating tokens outside vocabulary")
print("  3. Tokenizer can't decode these invalid token IDs")
print()
print("Root cause likely:")
print("  - Dequantization scales are wrong (too large or too small)")
print("  - INT8 → FP16 conversion losing precision")
print("  - Logits are wildly off, producing invalid token IDs")
print()
print("="*80)
print("RECOMMENDED FIX")
print("="*80)
print()
print("Run the quantization round-trip test:")
print("  cd /workspace/CodecLLM/core")
print("  python3 test_quantization_roundtrip.py")
print()
print("This will show if:")
print("  - Scales are reasonable")
print("  - INT8 reconstruction is accurate")
print("  - Dequantization math is correct")
print()
print("If that passes, the issue is likely:")
print("  - Layer normalization being compressed (should skip)")
print("  - Bias terms corrupted")
print("  - Activation functions affected")
print()

