#!/bin/bash

# Drop-in replacement for `sbatch run_escnn_batch.bash`: submits the same job, then
# captures the job ID sbatch prints and records it automatically via
# config_files/record_job_id.bash - no copy-pasting the ID by hand.

#   ./submit_and_record.bash            evaluate (default; --train does the same)
#   ./submit_and_record.bash --track    ekf
#   ./submit_and_record.bash --drift    run_multiple_eval

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "${1:-}" in
  ""|--train|--track|--drift) ;;
  *) echo "ERROR: unknown option '$1' (use --track, --drift, --train or nothing) - nothing submitted" >&2
     exit 1 ;;
esac

sbatch_output=$(sbatch "$script_dir/run_escnn_batch.bash" "$@")
echo "$sbatch_output"

job_id=$(printf '%s\n' "$sbatch_output" | grep -oE '[0-9]+' | tail -1)
if [ -z "$job_id" ]; then
  echo "WARNING: couldn't parse a job ID out of sbatch's output - nothing recorded" >&2
  exit 1
fi

"$script_dir/config_files/record_job_id.bash" "$job_id"
