#!/bin/bash

# Check for input file argument
if [ $# -ne 1 ]; then
  echo "Usage: $0 <input_file>"
  exit 1
fi

input_file=$1
base_name=$(basename "$input_file" .yaml)

# ---------------- Parameters ----------------
#seeds=(123 17 41 58)
seeds=(17)
snrs=($(seq -5 20))
cfos=(0)

clip_percentage_in_tx_vals=(100)
use_film_vals=(False)

# increase_prime_modulation sweep
increase_prime_modulation_vals=(False True)

# spatial_correlation sweep
spatial_correlation_vals=('none' 'medium')

# NEW: batch_size sweep
batch_size_vals=(1024)

# FIX: each augment mode must be a separate array element
which_augment_vals=(
  'AUGMENT_SPHERE'
  'AUGMENT_LMMSE'
)

TDL_model_vals=('C')
kernel_size_vals=(3)
run_tdfdcnn_vals=(False)

pilot_size_vals=(5000)
mcs_vals=(28 30 17)
override_noise_var_vals=(False)

# mod_pilot values (including negative)
mod_pilot_vals=(-1)

# n_users values
n_users_vals=(4)

# make_64QAM_16QAM_percentage values
make_64QAM_16QAM_percentage_vals=(0)

# --------------------------------------------
total_count=0
all_config_files=()

for seed in "${seeds[@]}"; do
  for cfo in "${cfos[@]}"; do
    echo "Generating configs for seed=$seed, cfo=$cfo"

    for clip in "${clip_percentage_in_tx_vals[@]}"; do
      for use_film in "${use_film_vals[@]}"; do
        for which_aug in "${which_augment_vals[@]}"; do
          for run_tdfdcnn in "${run_tdfdcnn_vals[@]}"; do

            [[ "$run_tdfdcnn" == True ]] && tdtag="td1" || tdtag="td0"

            for override_noise_var in "${override_noise_var_vals[@]}"; do
              [[ "$override_noise_var" == True ]] && ovtag="ov1" || ovtag="ov0"

              for tdl in "${TDL_model_vals[@]}"; do
                for spatial_corr in "${spatial_correlation_vals[@]}"; do

                  case "$spatial_corr" in
                    none)   sctag="sc0" ;;
                    medium) sctag="scM" ;;
                    high)   sctag="scH" ;;
                    *)      sctag="sc${spatial_corr}" ;;
                  esac

                  [[ "$use_film" == True ]] && uf="f1" || uf="f0"
                  ttag="T${tdl}"

                  case "$which_aug" in
                    NO_AUGMENT)      aug="NOAUG" ;;
                    AUGMENT_LMMSE)   aug="LMMSE" ;;
                    AUGMENT_DEEPRX)  aug="DEEPRX" ;;
                    AUGMENT_SPHERE)  aug="SPHERE" ;;
                    AUGMENT_DEEPSIC) aug="DEEPSIC" ;;
                    *)
                      echo "ERROR: Unknown which_augment: $which_aug" >&2
                      exit 1
                      ;;
                  esac

                  for kernel_size in "${kernel_size_vals[@]}"; do
                    ktag="k${kernel_size}"

                    for pilot_size in "${pilot_size_vals[@]}"; do
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

                        for n_users in "${n_users_vals[@]}"; do
                          utag="u${n_users}"

                          for mix_pct in "${make_64QAM_16QAM_percentage_vals[@]}"; do
                            mixtag="mix${mix_pct}"

                            for mod_pilot in "${mod_pilot_vals[@]}"; do
                              if [[ "$mod_pilot" -lt 0 ]]; then
                                mptag="mpm${mod_pilot#-}"
                              else
                                mptag="mp${mod_pilot}"
                              fi

                              for ipm_val in "${increase_prime_modulation_vals[@]}"; do
                                [[ "$ipm_val" == True ]] && ipm_tag="ipm1" || ipm_tag="ipm0"

                                for batch_size in "${batch_size_vals[@]}"; do
                                  bstag="bs${batch_size}"

                                  for snr in "${snrs[@]}"; do

                                    out_file="${base_name}_cfo${cfo}_clip${clip}_${uf}_${aug}_${ttag}_${sctag}_${ktag}_${ptag}_${mtag}_${utag}_${mptag}_${mixtag}_${ipm_tag}_${bstag}_${ovtag}_${tdtag}_s${seed}_snr${snr}.yaml"

                                    sed -e "s/^channel_seed:.*/channel_seed: $seed/" \
                                        -e "s/^snr:.*/snr: $snr/" \
                                        -e "s/^cfo:.*/cfo: $cfo/" \
                                        -e "s/^clip_percentage_in_tx:.*/clip_percentage_in_tx: $clip/" \
                                        -e "s/^use_film:.*/use_film: $use_film/" \
                                        -e "s/^which_augment:.*/which_augment: '$which_aug'/" \
                                        -e "s/^TDL_model:.*/TDL_model: '$tdl'/" \
                                        -e "s/^spatial_correlation:.*/spatial_correlation: '$spatial_corr'/" \
                                        -e "s/^kernel_size:.*/kernel_size: $kernel_size/" \
                                        -e "s/^run_tdfdcnn:.*/run_tdfdcnn: $run_tdfdcnn/" \
                                        -e "s/^pilot_size:.*/pilot_size: $pilot_size/" \
                                        -e "s/^mcs:.*/mcs: $mcs/" \
                                        -e "s/^n_users:.*/n_users: $n_users/" \
                                        -e "s/^mod_pilot:.*/mod_pilot: $mod_pilot/" \
                                        -e "s/^make_64QAM_16QAM_percentage:.*/make_64QAM_16QAM_percentage: $mix_pct/" \
                                        -e "s/^override_noise_var:.*/override_noise_var: $override_noise_var/" \
                                        -e "s/^increase_prime_modulation:.*/increase_prime_modulation: $ipm_val/" \
                                        -e "s/^batch_size:.*/batch_size: $batch_size/" \
                                        "$input_file" > "$out_file"

                                    all_config_files+=("$out_file")
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
              done
            done
          done
        done
      done
    done

    echo "Finished seed=$seed cfo=$cfo"
  done
done

# ----- PRINTING: match old version output style -----
quoted_files=()
for f in "${all_config_files[@]}"; do
  safe_f=${f//\"/\\\"}
  quoted_files+=("\\\"$safe_f\\\"")
done

config_line="config_files=(${quoted_files[*]})"
echo "\"$config_line\""

echo "Total configs generated: $total_count"

