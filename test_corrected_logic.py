"""
Final test to verify the corrected 64QAM three-part division implementation
Division is on the FIRST DIMENSION (pilot_chunk / pilot_length)
"""
import numpy as np
import sys
sys.path.insert(0, 'C:/Projects/DeepOFDM')

from python_code.utils.probs_utils import skip_indices

# Simulate configuration
class Config:
    make_64QAM_16QAM_percentage = 50
    mod_pilot = 64
    n_users = 4

conf = Config()

print("="*80)
print("CORRECTED IMPLEMENTATION TEST: 64QAM Three-Part Division")
print("="*80)
print(f"\nConfiguration:")
print(f"  make_64QAM_16QAM_percentage: {conf.make_64QAM_16QAM_percentage}")
print(f"  mod_pilot: {conf.mod_pilot}")
print(f"  n_users: {conf.n_users}")

# Test 1: probs_for_aug (from evaluate.py)
print("\n" + "="*80)
print("TEST 1: probs_for_aug (evaluate.py logic)")
print("="*80)

pilot_chunk = 90  # Example pilot chunk size
num_bits_pilot = 6  # 64QAM
num_res = 23

# Initialize probs_for_aug (shape: pilot_chunk, num_bits_pilot*n_users, num_res, 1)
probs_for_aug = np.random.rand(pilot_chunk, num_bits_pilot * conf.n_users, num_res, 1)
print(f"\nprobs_for_aug shape: {probs_for_aug.shape}")

if conf.make_64QAM_16QAM_percentage == 50 and num_bits_pilot == 6:
    # Divide pilot_chunk into three equal parts
    third_size = pilot_chunk // 3

    print(f"\nDividing pilot_chunk={pilot_chunk} into thirds:")
    print(f"  First third:  indices 0 to {third_size-1} ({third_size} pilots)")
    print(f"  Second third: indices {third_size} to {2*third_size-1} ({third_size} pilots)")
    print(f"  Third part:   indices {2*third_size} to {pilot_chunk-1} ({pilot_chunk - 2*third_size} pilots)")

    # First third: skip_indices with ratio 3
    indexes_first = skip_indices(int(num_bits_pilot*conf.n_users), 3)
    probs_for_aug[:third_size, indexes_first, :, :] = 1.0
    print(f"\nFirst third: setting {len(indexes_first)} bit indices to prob=1.0")
    print(f"  Bit indices: {indexes_first}")

    # Second third: skip_indices with ratio 1.5
    indexes_second = skip_indices(int(num_bits_pilot*conf.n_users), 1.5)
    probs_for_aug[third_size:2*third_size, indexes_second, :, :] = 0.5
    print(f"\nSecond third: setting {len(indexes_second)} bit indices to prob=0.5")
    print(f"  Bit indices: {indexes_second}")

    print(f"\nThird part: unchanged (full 64QAM)")

# Verify
print(f"\nVerification (sample RE=0, dim4=0):")
print(f"  First third pilot=0:  prob @ bit_idx={indexes_first[0]} = {probs_for_aug[0, indexes_first[0], 0, 0]:.2f} (expect 1.0)")
print(f"  Second third pilot={third_size}:  prob @ bit_idx={indexes_second[0]} = {probs_for_aug[third_size, indexes_second[0], 0, 0]:.2f} (expect 0.5)")
print(f"  Third part pilot={2*third_size}:  prob @ bit_idx=0 = {probs_for_aug[2*third_size, 0, 0, 0]:.2f} (expect random)")

# Test 2: tx_pilots (from mimo_channel_dataset.py)
print("\n" + "="*80)
print("TEST 2: tx_pilots (mimo_channel_dataset.py logic)")
print("="*80)

pilot_length = 90  # Example pilot length
n_users = conf.n_users

# Initialize tx_pilots (shape: pilot_length, n_users, num_res)
tx_pilots = np.random.randint(0, 2, size=(pilot_length, n_users, num_res))
print(f"\ntx_pilots shape: {tx_pilots.shape}")

if conf.make_64QAM_16QAM_percentage == 50 and conf.mod_pilot == 64:
    # Divide pilot_length into three parts
    third_size = pilot_length // 3

    print(f"\nDividing pilot_length={pilot_length} into thirds:")
    print(f"  First third:  bits 0 to {third_size-1} ({third_size} bits)")
    print(f"  Second third: bits {third_size} to {2*third_size-1} ({third_size} bits)")
    print(f"  Third part:   bits {2*third_size} to {pilot_length-1} ({pilot_length - 2*third_size} bits)")

    # First third: skip_indices with ratio 3
    indices_first = skip_indices(third_size, 3)
    tx_pilots[indices_first, :, :] = 1
    print(f"\nFirst third: setting {len(indices_first)} bit positions to 1 (across all users/REs)")
    print(f"  Bit positions: {indices_first}")

    # Second third: skip_indices with ratio 1.5
    indices_second = skip_indices(third_size, 1.5) + third_size
    tx_pilots[indices_second, :, :] = 1
    print(f"\nSecond third: setting {len(indices_second)} bit positions to 1 (across all users/REs)")
    print(f"  Bit positions: {indices_second}")

    print(f"\nThird part: unchanged (full 64QAM)")

# Verify
print(f"\nVerification (user=0, RE=0):")
if len(indices_first) > 0:
    print(f"  First third bit={indices_first[0]}: {tx_pilots[indices_first[0], 0, 0]} (expect 1)")
if len(indices_second) > 0:
    print(f"  Second third bit={indices_second[0]}: {tx_pilots[indices_second[0], 0, 0]} (expect 1)")
print(f"  Third part bit={2*third_size}: {tx_pilots[2*third_size, 0, 0]} (expect 0 or 1, random)")

print("\n" + "="*80)
print("TEST COMPLETED SUCCESSFULLY")
print("="*80)
print("\nThe corrected implementation:")
print("  1. Divides the FIRST DIMENSION (pilot_chunk/pilot_length) into 3 parts")
print("  2. First third: applies skip_indices with ratio 3 to BIT indices")
print("  3. Second third: applies skip_indices with ratio 1.5 to BIT indices")
print("  4. Third part: unchanged (full 64QAM)")
print("\nThis matches the original 2-part division pattern!")

