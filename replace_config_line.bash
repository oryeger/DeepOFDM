#!/bin/bash

# Check for one argument
if [ $# -ne 1 ]; then
  echo "Usage: $0 \"config_files=(file1.yaml file2.yaml ...)\""
  exit 1
fi

new_line="$1"
target_file="run_deepsic_batch.bash"

# Escape slashes for sed
escaped_line=$(printf '%s\n' "$new_line" | sed 's/[&/\]/\\&/g')

# Replace the line in-place
sed -i -E "s|^config_files=\(.*\)$|$escaped_line|" "$target_file"

echo "Replaced config_files line in $target_file"

