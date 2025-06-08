#!/bin/bash

# Check for input file argument
if [ $# -ne 1 ]; then
  echo "Usage: $0 <input_file>"
  exit 1
fi

input_file=$1
base_name=$(basename "$input_file" .yaml)

# Define seeds
seeds=(41 123 17 58)

total_count=0
all_config_files=()

for seed in "${seeds[@]}"; do
  echo "Generating configs for seed $seed"

  for i in $(seq 5 30); do
    out_file="${base_name}_seed${seed}_${i}.yaml"

    # Replace channel_seed and snr lines
    sed -e "s/^channel_seed:.*/channel_seed: $seed/" -e "s/^snr:.*/snr: $i/" "$input_file" > "$out_file"

    all_config_files+=("\"$out_file\"")
    ((total_count++))
  done

  echo "Generated 26 config files for seed $seed"
  echo ""
done

# Print full config_files array
echo "config_files=(${all_config_files[*]})"
echo "Total config files generated: $total_count"

