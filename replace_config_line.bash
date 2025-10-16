#!/bin/bash

# Check for one argument
if [ $# -ne 1 ]; then
  echo "Usage: $0 \"config_files=(file1.yaml file2.yaml ...)\""
  exit 1
fi

new_line="$1"
target_file="run_escnn_batch.bash"

# Escape slashes for sed
escaped_line=$(printf '%s\n' "$new_line" | sed 's/[&/\]/\\&/g')

# Replace the line in-place
sed -i -E "s|^config_files=\(.*\)$|$escaped_line|" "$target_file"

echo "Replaced config_files line in $target_file"

# ---------------- Add this part ----------------
# Count number of files between parentheses
inside_parens=$(printf '%s\n' "$new_line" | sed -E 's/^config_files=\(|\)$//g')
num_files=$(printf '%s\n' "$inside_parens" | wc -w | awk '{print $1}')
last_idx=$(( num_files - 1 ))

# Update the SBATCH array line
sed -i -E "s|^#SBATCH[[:space:]]+--array=0-[0-9]+|#SBATCH --array=0-${last_idx}|" "$target_file"

echo "Updated #SBATCH --array=0-${last_idx} in $target_file"

