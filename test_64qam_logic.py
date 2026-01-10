"""
Test script to verify the 64QAM to 16QAM three-part division logic
"""
import numpy as np
from python_code.utils.probs_utils import get_64QAM_16QAM_indices_and_probs, skip_indices

# Test with 64QAM configuration
bits_per_symbol = 6  # 64QAM
n_users = 4
total_bits = bits_per_symbol * n_users  # 24 bits total

print(f"Testing with {n_users} users and {bits_per_symbol} bits per symbol (64QAM)")
print(f"Total bits: {total_bits}")
print()

# Get the three-part division
first_third_indices, second_third_indices, first_third_size, second_third_size, third_size = \
    get_64QAM_16QAM_indices_and_probs(total_bits, bits_per_symbol=6)

print(f"First third size: {first_third_size} bits ({first_third_size // bits_per_symbol} symbols)")
print(f"Second third size: {second_third_size} bits ({second_third_size // bits_per_symbol} symbols)")
print(f"Third part size: {third_size} bits ({third_size // bits_per_symbol} symbols)")
print()

print(f"First third indices (skip_indices with ratio 3): {first_third_indices}")
print(f"Number of indices in first third: {len(first_third_indices)}")
print()

print(f"Second third indices (skip_indices with ratio 1.5): {second_third_indices}")
print(f"Number of indices in second third: {len(second_third_indices)}")
print()

# Verify that the indices don't overlap
overlap = np.intersect1d(first_third_indices, second_third_indices)
print(f"Overlap between first and second thirds: {overlap} (should be empty)")
print()

# Verify the distribution
all_indices = np.arange(total_bits)
third_part_start = first_third_size + second_third_size
third_part_indices = np.arange(third_part_start, total_bits)

print(f"Third part indices (unchanged): {third_part_indices}")
print(f"Number of indices in third part: {len(third_part_indices)}")
print()

# Summary
print("Summary:")
print(f"- First third ({first_third_size // bits_per_symbol} symbols): {len(first_third_indices)} bits set to 1 (prob=1.0)")
print(f"- Second third ({second_third_size // bits_per_symbol} symbols): {len(second_third_indices)} bits set to 1 (prob=0.5)")
print(f"- Third part ({third_size // bits_per_symbol} symbols): unchanged (full 64QAM)")

