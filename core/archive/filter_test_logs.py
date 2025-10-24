#!/usr/bin/env python3
"""
Filter test logs to remove repetitive encoder/decoder messages
Usage: python test_all_layers_compressed.py 2>&1 | python filter_test_logs.py
"""

import sys
import re

# Patterns to suppress (keep only first occurrence)
suppress_after_first = [
    r'\[ENCODER\] Starting nvCOMP',
    r'\[ENCODER\] Temp:',
    r'\[ENCODER\] Compressed size:',
    r'\[ENCODER\] .* GPU compression SUCCESS',
    r'\[DECODER\] .* GPU direct decode SUCCESS',
    r'\[DECODER DEBUG\] nvcompStatus=0',
]

# Patterns to completely suppress
always_suppress = [
    r'`torch_dtype` is deprecated',
]

# Count occurrences of each pattern
seen_patterns = {}

print("="*80)
print("FILTERED TEST OUTPUT")
print("="*80)
print()

for line in sys.stdin:
    line = line.rstrip()
    
    # Check if should always suppress
    should_suppress = False
    for pattern in always_suppress:
        if re.search(pattern, line):
            should_suppress = True
            break
    
    if should_suppress:
        continue
    
    # Check if should suppress after first occurrence
    suppress_now = False
    for pattern in suppress_after_first:
        if re.search(pattern, line):
            if pattern in seen_patterns:
                # Show every 50th occurrence for progress
                seen_patterns[pattern] += 1
                if seen_patterns[pattern] % 50 == 0:
                    print(f"[... {seen_patterns[pattern]} similar messages ...]")
                suppress_now = True
            else:
                seen_patterns[pattern] = 1
                # Show first occurrence
            break
    
    if not suppress_now:
        print(line)

# Summary at end
print()
print("="*80)
print("LOG SUMMARY")
print("="*80)
for pattern, count in seen_patterns.items():
    if count > 1:
        # Extract pattern description
        if 'ENCODER' in pattern:
            msg_type = "Encoder messages"
        elif 'DECODER' in pattern:
            msg_type = "Decoder messages"
        else:
            msg_type = "Messages"
        print(f"  {msg_type}: {count} occurrences (showing 1/{count})")

