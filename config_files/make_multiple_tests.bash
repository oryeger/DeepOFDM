#!/bin/bash

# Check for input file argument
if [ $# -ne 1 ]; then
  echo "Usage: $0 <input_file>"
  exit 1
fi

input_file=$1
base_name=$(basename "$input_file" .yaml)

# Define parameters
seeds=(123)
snrs=($(seq 5 30))
cfos=(0)
clip_percentage_in_tx_vals=(100 37 34)
use_film_vals=(False True)
which_augment_vals=('NO_AUGMENT')
TDL_model_vals=('C' 'N')   # NEW

total_count=0
all_config_files=()

for seed in "${seeds[@]}"; do
  for cfo in "${cfos[@]}"; do
    echo "Generating configs for seed=$seed, cfo=$cfo"

    for clip in "${clip_percentage_in_tx_vals[@]}"; do
      for use_film in "${use_film_vals[@]}"; do
        for which_aug in "${which_augment_vals[@]}"; do
          for tdl in "${TDL_model_vals[@]}"; do   # NEW LOOP

            # ---- Build SHORT TAGS ----
            [[ "$use_film" == True ]] && uf="f1" || uf="f0"
            [[ "$which_aug" == "NO_AUGMENT" ]] && aug="NA" || aug="LMMSE"
            ttag="T${tdl}"   # e.g. TC / TN in filename

            for snr in "${snrs[@]}"; do

              # Filename with seed + snr at the END
              # now also includes TDL tag
              out_file="${base_name}_cfo${cfo}_clip${clip}_${uf}_${aug}_${ttag}_s${seed}_snr${snr}.yaml"

              # Replace YAML fields
              sed -e "s/^channel_seed:.*/channel_seed: $seed/" \
                  -e "s/^snr:.*/snr: $snr/" \
                  -e "s/^cfo:.*/cfo: $cfo/" \
                  -e "s/^clip_percentage_in_tx:.*/clip_percentage_in_tx: $clip/" \
                  -e "s/^use_film:.*/use_film: $use_film/" \
                  -e "s/^which_augment:.*/which_augment: '$which_aug'/" \
                  -e "s/^TDL_model:.*/TDL_model: '$tdl'/" \
                  "$input_file" > "$out_file"

              all_config_files+=("\"$out_file\"")
              ((total_count++))
            done
          done
        done
      done
    done

    echo "Finished seed=$seed cfo=$cfo"
  done
done

# Build config_files array
quoted_files=()
for f in "${all_config_files[@]}"; do
  quoted_files+=("\\\"$f\\\"")
done

config_line="config_files=(${quoted_files[*]})"
echo "\"$config_line\""

