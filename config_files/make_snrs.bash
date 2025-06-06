#!/bin/bash

# Check for input file argument
if [ $# -ne 1 ]; then
  echo "Usage: $0 <input_file>"
  exit 1
fi

input_file=$1
base_name=$(basename "$input_file" .yaml)

for i in $(seq 5 25); do
  out_file="${base_name}_${i}.yaml"
  {
    echo "snr: $i"
    tail -n +2 "$input_file"
  } > "$out_file"
done

echo "✅ Generated files: ${base_name}_1dB.yaml to ${base_name}_20dB.yaml"

