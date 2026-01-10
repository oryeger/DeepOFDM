"""
Test script with larger pilot size to verify the 64QAM to 16QAM three-part division logic
"""
import numpy as np
from python_code.utils.probs_utils import get_64QAM_16QAM_indices_and_probs, skip_indices

# Test with 64QAM configuration and larger pilot
bits_per_symbol = 6  # 64QAM
n_users = 4

# Test with different pilot sizes
for pilot_symbols_per_user in [3, 6, 9, 12, 15]:
    total_bits = bits_per_symbol * n_users * pilot_symbols_per_user

    print(f"\n{'='*70}")
    print(f"Testing with {n_users} users, {pilot_symbols_per_user} symbols per user")
    print(f"Total bits: {total_bits} ({total_bits // bits_per_symbol} symbols total)")
    print(f"{'='*70}")

    # Get the three-part division
    first_third_indices, second_third_indices, first_third_size, second_third_size, third_size = \
        get_64QAM_16QAM_indices_and_probs(total_bits, bits_per_symbol=6)

    print(f"\nFirst third: {first_third_size} bits ({first_third_size // bits_per_symbol} symbols)")
    print(f"  - Skip indices (ratio 3): {len(first_third_indices)} bits set to 1")
    print(f"  - Probability: 1.0 (certain)")

    print(f"\nSecond third: {second_third_size} bits ({second_third_size // bits_per_symbol} symbols)")
    print(f"  - Skip indices (ratio 1.5): {len(second_third_indices)} bits set to 1")
    print(f"  - Probability: 0.5 (augmentation)")

    print(f"\nThird part: {third_size} bits ({third_size // bits_per_symbol} symbols)")
    print(f"  - Unchanged (full 64QAM)")

    # Verify total
    total_check = first_third_size + second_third_size + third_size
    print(f"\nTotal check: {total_check} bits (should equal {total_bits})")
    assert total_check == total_bits, "Total bits mismatch!"

    # Show bit distribution
    print(f"\nBit distribution:")
    print(f"  - First third: bits 0-{first_third_size-1}")
    print(f"  - Second third: bits {first_third_size}-{first_third_size + second_third_size - 1}")
    print(f"  - Third part: bits {first_third_size + second_third_size}-{total_bits-1}")

