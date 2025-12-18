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
snrs=($(seq 5 30))
cfos=(0)
clip_percentage_in_tx_vals=(100)
use_film_vals=(False)
which_augment_vals=('NO_AUGMENT' 'AUGMENT_LMMSE')
TDL_model_vals=('C')     # NEW
kernel_size_vals=(3)           # NEW
run_tdfdcnn_vals=(False)          # NEW
pilot_size_vals=(5000 20000)     # NEW
mcs_vals=(4)                 # NEW
override_noise_var_vals=(False)  # NEW

total_count=0
all_config_files=()

for seed in "${seeds[@]}"; do
  for cfo in "${cfos[@]}"; do
    echo "Generating configs for seed=$seed, cfo=$cfo"

    for clip in "${clip_percentage_in_tx_vals[@]}"; do
      for use_film in "${use_film_vals[@]}"; do
        for which_aug in "${which_augment_vals[@]}"; do
          for run_tdfdcnn in "${run_tdfdcnn_vals[@]}"; do

            # TD tag for filename
            [[ "$run_tdfdcnn" == True ]] && tdtag="td1" || tdtag="td0"

            for override_noise_var in "${override_noise_var_vals[@]}"; do
              # override_noise_var tag
              [[ "$override_noise_var" == True ]] && ovtag="ov1" || ovtag="ov0"

              for tdl in "${TDL_model_vals[@]}"; do

                [[ "$use_film" == True ]] && uf="f1" || uf="f0"
                [[ "$which_aug" == "NO_AUGMENT" ]] && aug="NA" || aug="LMMSE"
                ttag="T${tdl}"

                for kernel_size in "${kernel_size_vals[@]}"; do
                  ktag="k${kernel_size}"

                  for pilot_size in "${pilot_size_vals[@]}"; do
                    # pilot tag for filename
                    if [[ "$pilot_size" -eq 1000 ]]; then
                      ptag="p1k"
                    elif [[ "$pilot_size" -eq 5000 ]]; then
                      ptag="p5k"
                    elif [[ "$pilot_size" -eq 10000 ]]; then
                      ptag="p10k"
                    elif [[ "$pilot_size" -eq 20000 ]]; then
                      ptag="p20k"
                    else
                      ptag="p${pilot_size}"
                    fi

                    for mcs in "${mcs_vals[@]}"; do
                      mtag="m${mcs}"

                      for snr in "${snrs[@]}"; do

                        # Filename including td tag, pilot tag, mcs tag, and override_noise_var tag
                        out_file="${base_name}_cfo${cfo}_clip${clip}_${uf}_${aug}_${ttag}_${ktag}_${ptag}_${mtag}_${ovtag}_${tdtag}_s${seed}_snr${snr}.yaml"

                        sed -e "s/^channel_seed:.*/channel_seed: $seed/" \
                            -e "s/^snr:.*/snr: $snr/" \
                            -e "s/^cfo:.*/cfo: $cfo/" \
                            -e "s/^clip_percentage_in_tx:.*/clip_percentage_in_tx: $clip/" \
                            -e "s/^use_film:.*/use_film: $use_film/" \
                            -e "s/^which_augment:.*/which_augment: '$which_aug'/" \
                            -e "s/^TDL_model:.*/TDL_model: '$tdl'/" \
                            -e "s/^kernel_size:.*/kernel_size: $kernel_size/" \
                            -e "s/^run_tdfdcnn:.*/run_tdfdcnn: $run_tdfdcnn/" \
                            -e "s/^pilot_size:.*/pilot_size: $pilot_size/" \
                            -e "s/^mcs:.*/mcs: $mcs/" \
                            -e "s/^override_noise_var:.*/override_noise_var: $override_noise_var/" \
                            "$input_file" > "$out_file"

                        all_config_files+=("\"$out_file\"")
                        ((total_count++))
                      done
                    done
                  done
                done
              done
            done
          done
        done
      done
    done

    echo "Finished seed=$seed cfo=$cfo"
  done
done

quoted_files=()
for f in "${all_config_files[@]}"; do
  quoted_files+=("\\\"$f\\\"")
done

config_line="config_files=(${quoted_files[*]})"
echo "\"$config_line\""

