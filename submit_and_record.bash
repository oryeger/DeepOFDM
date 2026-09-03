#!/bin/bash

# Drop-in replacement for `sbatch run_escnn_batch.bash`: submits the same job, then
# captures the job ID sbatch prints and records it automatically via
# config_files/record_job_id.bash - no copy-pasting the ID by hand.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

sbatch_output=$(sbatch "$script_dir/run_escnn_batch.bash" "$@")
echo "$sbatch_output"

job_id=$(printf '%s\n' "$sbatch_output" | grep -oE '[0-9]+' | tail -1)
if [ -z "$job_id" ]; then
  echo "WARNING: couldn't parse a job ID out of sbatch's output - nothing recorded" >&2
  exit 1
fi

"$script_dir/config_files/record_job_id.bash" "$job_id"
