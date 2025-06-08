#!/bin/bash

# Check for input file argument
if [ $# -ne 1 ]; then
  echo "Usage: $0 <input_file>"
  exit 1
fi

input_file=$1
base_name=$(basename "$input_file" .yaml)

# Initialize array
config_files=()

for i in $(seq 5 30); do
  out_file="${base_name}_${i}.yaml"
#  {
#    echo "snr: $i"
#    tail -n +2 "$input_file"
#  } > "$out_file"

  config_files+=("\"$out_file\"")
done

# Print the array in the required format
echo "config_files=(${config_files[*]})"

