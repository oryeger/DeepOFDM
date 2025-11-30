#!/bin/bash

# Check for input file argument
if [ $# -ne 1 ]; then
  echo "Usage: $0 <input_file>"
  exit 1
fi

input_file=$1
base_name=$(basename "$input_file" .yaml)

# Define parameters
seeds=(123 17 41 58)
snrs=($(seq 10 35))  # You can replace this with a custom list, e.g., snrs=(0 5 10 15)
cfos=(0)
clip_percentage_in_tx_vals=(100 37 34)

total_count=0
all_config_files=()

for seed in "${seeds[@]}"; do
  for cfo in "${cfos[@]}"; do
    echo "Generating configs for seed $seed, cfo $cfo"

    for clip in "${clip_percentage_in_tx_vals[@]}"; do
      echo "  -> clip_percentage_in_tx = $clip"

      for snr in "${snrs[@]}"; do
        out_file="${base_name}_seed${seed}_cfo${cfo//./p}_clip${clip}_snr${snr}.yaml"

        # Replace channel_seed, snr, cfo and clip_percentage_in_tx lines
        sed -e "s/^channel_seed:.*/channel_seed: $seed/" \
            -e "s/^snr:.*/snr: $snr/" \
            -e "s/^cfo:.*/cfo: $cfo/" \
            -e "s/^clip_percentage_in_tx:.*/clip_percentage_in_tx: $clip/" \
            "$input_file" > "$out_file"

        all_config_files+=("\"$out_file\"")
        ((total_count++))
      done
    done

    echo "Generated $((${#snrs[@]} * ${#clip_percentage_in_tx_vals[@]})) config files for seed $seed, cfo $cfo"
    echo ""
  done
done

# Print full config_files array line
quoted_files=()
for f in "${all_config_files[@]}"; do
  quoted_files+=("\\\"$f\\\"")
done

config_line="config_files=(${quoted_files[*]})"
echo "\"$config_line\""

