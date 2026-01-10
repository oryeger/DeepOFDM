# Implementation Summary: 64QAM to 16QAM Three-Part Division

## Overview
This implementation updates the `make_64QAM_16QAM_percentage` configuration to support a new mode when set to `50` for 64QAM modulation. Instead of a simple percentage-based modification, it now divides the pilot signals into three equal parts along the **first dimension** (pilot_chunk / pilot_length) with different skip_indices configurations.

## Changes Made

### 1. Updated Training Logic (`evaluate.py`)
Modified the online training section to:
- Detect when `make_64QAM_16QAM_percentage == 50` and `num_bits_pilot == 6` (64QAM)
- Divide `pilot_chunk` (first dimension) into three equal parts
- Apply the three-part logic:
  - **First third** (pilots 0 to pilot_chunk//3): Apply skip_indices with ratio 3 to bit indices, set probabilities to 1.0
  - **Second third** (pilots pilot_chunk//3 to 2*pilot_chunk//3): Apply skip_indices with ratio 1.5 to bit indices, set probabilities to 0.5
  - **Third part** (pilots 2*pilot_chunk//3 to pilot_chunk): Unchanged (full 64QAM)
- Maintains backward compatibility with the original logic for other percentage values

**Location**: `C:\Projects\DeepOFDM\python_code\evaluate.py` (around line 614)

### 2. Updated Channel Dataset (`mimo_channel_dataset.py`)
Modified the transmit function to:
- Apply the same three-part division to `tx_pilots` along the first dimension
- Divide `pilot_length` into three equal parts
- For each part, apply skip_indices to determine which bit positions to set to 1
  - **First third**: skip_indices with ratio 3
  - **Second third**: skip_indices with ratio 1.5 (offset by third_size)
  - **Third part**: unchanged
- Maintains backward compatibility with the original logic for other percentage values

**Location**: `C:\Projects\DeepOFDM\python_code\channel\mimo_channels\mimo_channel_dataset.py` (around line 60)

## Logic Explanation

### Division Strategy
The first dimension is divided into three equal parts:
- First third: `0` to `size // 3 - 1`
- Second third: `size // 3` to `2 * size // 3 - 1`
- Third part: `2 * size // 3` to `size - 1`

### Pattern Match with Original 2-Part Division
The original code for 2-part division:
```python
indexes = skip_indices(int(num_bits_pilot*conf.n_users), pilot_data_ratio)
probs_for_aug[:int(pilot_chunk*conf.make_64QAM_16QAM_percentage/100), indexes, :, :] = 0.5
```

The new 3-part division follows the same pattern:
```python
# Divide first dimension into thirds
third_size = pilot_chunk // 3

# First third: skip_indices with ratio 3 on bit dimension
indexes_first = skip_indices(int(num_bits_pilot*conf.n_users), 3)
probs_for_aug[:third_size, indexes_first, :, :] = 1.0

# Second third: skip_indices with ratio 1.5 on bit dimension
indexes_second = skip_indices(int(num_bits_pilot*conf.n_users), 1.5)
probs_for_aug[third_size:2*third_size, indexes_second, :, :] = 0.5

# Third part: unchanged
```

### Bit Manipulation in tx_pilots
For `tx_pilots` with shape `(pilot_length, n_users, num_res)`:

```python
# Divide first dimension into thirds
pilot_length = tx_pilots.shape[0]
third_size = pilot_length // 3

# First third: skip_indices with ratio 3
indices_first = skip_indices(third_size, 3)
tx_pilots[indices_first, :, :] = 1

# Second third: skip_indices with ratio 1.5
indices_second = skip_indices(third_size, 1.5) + third_size
tx_pilots[indices_second, :, :] = 1

# Third part: unchanged
```

## Example Output
For pilot_chunk=90 with 4 users and 6 bits per symbol:

**probs_for_aug (shape: 90, 24, 23, 1)**:
```
First third:  indices 0-29 (30 pilots)
  - 16 bit indices set to prob=1.0 (skip_indices with ratio 3 on 24 bits)
Second third: indices 30-59 (30 pilots)
  - 8 bit indices set to prob=0.5 (skip_indices with ratio 1.5 on 24 bits)
Third part:   indices 60-89 (30 pilots)
  - Unchanged (full 64QAM)
```

**tx_pilots (shape: 90, 4, 23)**:
```
First third:  bits 0-29 (30 bits)
  - 20 bit positions set to 1 (skip_indices with ratio 3 on 30 bits)
Second third: bits 30-59 (30 bits)
  - 10 bit positions set to 1 (skip_indices with ratio 1.5 on 30 bits)
Third part:   bits 60-89 (30 bits)
  - Unchanged (full 64QAM)
```

## Testing
Test script created to verify the implementation:
- `test_corrected_logic.py`: Verifies the correct division pattern

Test confirms:
- Correct division of first dimension into three parts
- Proper application of skip_indices to bit dimension
- Probabilities set correctly
- Pattern matches original 2-part division

## Backward Compatibility
The implementation maintains full backward compatibility:
- When `make_64QAM_16QAM_percentage != 50`, the original logic is used
- When modulation is not 64QAM, the original logic is used
- Existing configurations continue to work without modification

## Files Modified
1. `python_code/evaluate.py` - Updated training logic
2. `python_code/channel/mimo_channels/mimo_channel_dataset.py` - Updated channel dataset logic

## Files Created (for testing)
1. `test_corrected_logic.py` - Verification test script

