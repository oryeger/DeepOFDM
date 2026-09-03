#!/bin/bash

# Run this a few seconds after `sbatch run_escnn_batch.bash`, passing the printed job ID
# (either just the number, or the whole "Submitted batch job 123456" line - either works).
# Appends that job ID to the filenames of the two most-recently archived files from the
# last make_multiple_tests.bash run (config_files/archive/<ts>_make_<cur_str>.bash and
# <ts>_mult_<cur_str>.yaml) - e.g. <ts>_make_<cur_str>_job123456.bash - so a past archived
# batch stays traceable to the SLURM job(s) that actually ran it, without touching the
# archived files' contents. Safe to call again for a resubmit: a new job ID appends
# alongside any already recorded, and re-recording the same ID is a no-op.

if [ $# -ne 1 ]; then
  echo "Usage: $0 <job_id | \"Submitted batch job <job_id>\">"
  exit 1
fi

job_id=$(printf '%s\n' "$1" | grep -oE '[0-9]+' | tail -1)
if [ -z "$job_id" ]; then
  echo "ERROR: couldn't find a job ID in '$1'" >&2
  exit 1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
archive_dir="$script_dir/archive"

if [ ! -d "$archive_dir" ]; then
  echo "ERROR: no archive directory at $archive_dir yet - run make_multiple_tests.bash first" >&2
  exit 1
fi

latest_script=$(ls -1 "$archive_dir"/*_make_*.bash 2>/dev/null | sort | tail -1)
latest_yaml=$(ls -1 "$archive_dir"/*_mult_*.yaml 2>/dev/null | sort | tail -1)

if [ -z "$latest_script" ] && [ -z "$latest_yaml" ]; then
  echo "ERROR: no archived *_make_*.bash or *_mult_*.yaml files found under $archive_dir" >&2
  exit 1
fi

for target in "$latest_script" "$latest_yaml"; do
  [ -z "$target" ] && continue
  dir=$(dirname "$target")
  base=$(basename "$target")
  ext="${base##*.}"
  stem="${base%.*}"
  if printf '%s' "$stem" | grep -qE "(^|_)job${job_id}(_|\$)"; then
    echo "Job $job_id already recorded in $target - skipping"
    continue
  fi
  new_target="$dir/${stem}_job${job_id}.${ext}"
  mv "$target" "$new_target"
  echo "Recorded job $job_id: renamed $base -> $(basename "$new_target")"
done
