#!/usr/bin/env python3
"""List the archived generator scripts of the Slurm jobs that are still in the queue.

Usage (on the cluster):
    ./running_jobs_archive.py
    ./running_jobs_archive.py --user egero --archive /home/egero/Projects/DeepOFDM/config_files/archive

Takes the job IDs from `squeue -u <user> -h -r -o "%F %T"` (array parent ID + task state;
-r expands pending array ranges like 23349678_[10-305] into one line per task, so the
counts are per task, not per squeue display line), then finds the
config_files/archive/*.bash files whose name carries that ID. The match uses the same
convention record_job_id.bash writes and checks: the token `job<ID>` delimited by '_' or
end of stem, e.g.
    <ts>_make_<cur_str>_job23349678.bash
    <ts>_make_<cur_str>_job23349678_job23357045.bash   (resubmit appends a second ID)
so job 2334967 does not falsely match job23349678.

Output: first one `vi <full path>` line per matched file (ready to copy-paste), then each
queued job ID with its cur_str (e.g. icpilot) and running / pending / other task counts,
then any job IDs that have no archived file.
"""
import argparse
import getpass
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict

# python_code/utils/ -> repo root -> config_files/archive
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DEFAULT_ARCHIVE = os.path.join(_REPO_ROOT, 'config_files', 'archive')
FALLBACK_ARCHIVE = '/home/egero/Projects/DeepOFDM/config_files/archive'

# <ts>_make_<cur_str>[_job<ID>...] - ts is make_multiple_tests.bash's `date +%Y%m%d_%H%M`,
# the _job<ID> suffixes are appended by record_job_id.bash
_ARCHIVE_NAME = re.compile(r'^\d{8}_\d{4}_make_(?P<cur_str>.+?)(?:_job\d+)*$')


def queued_jobs(user):
    """{array-parent job ID (%F): Counter of task states}, for the user's queued jobs."""
    try:
        out = subprocess.run(['squeue', '-u', user, '-h', '-r', '-o', '%F %T'],
                             capture_output=True, text=True, check=True).stdout
    except FileNotFoundError:
        sys.exit('ERROR: squeue not found - run this on the Slurm login node')
    except subprocess.CalledProcessError as e:
        sys.exit(f'ERROR: squeue failed: {e.stderr.strip()}')
    jobs = defaultdict(Counter)
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0].isdigit():
            jobs[parts[0]][parts[1]] += 1
    return dict(sorted(jobs.items(), key=lambda kv: int(kv[0])))


def archived_scripts_for(job_id, archive_dir, bash_files):
    pat = re.compile(rf'(^|_)job{job_id}(_|$)')
    return [os.path.join(archive_dir, f) for f in bash_files
            if pat.search(os.path.splitext(f)[0])]


_LOOP_ARRAY = re.compile(r'^\s*for\s+\w+\s+in\s+"\$\{(\w+)\[@\]\}"', re.M)


def total_tests(path):
    """Number of configs (= array tasks) the archived make_multiple_tests.bash generated: the
    product of the lengths of the arrays its loops run over (the loops never skip a combination).
    The Parameters section (everything before `total_count=0`) is run in bash to evaluate the
    arrays - it only assigns variables, e.g. snrs=($(seq 0 1 30)). None if that fails."""
    try:
        with open(path) as f:
            text = f.read()
        head, sep, body = text.partition('\ntotal_count=0')
        body = body.partition('((total_count++))')[0]   # later loops (all_config_files) aren't sweeps
        names = list(dict.fromkeys(_LOOP_ARRAY.findall(body)))
        if not sep or not names:
            return None
        script = (head + '\nn=1\nfor a in ' + ' '.join(names) +
                  '; do eval "k=\\${#$a[@]}"; n=$((n*k)); done\necho $n\n')
        # '-- x.yaml' satisfies the script's one-argument check
        out = subprocess.run(['bash', '-s', '--', 'x.yaml'], input=script, capture_output=True,
                             text=True, timeout=10).stdout.split()
        return int(out[-1]) if out and out[-1].isdigit() else None
    except (OSError, subprocess.SubprocessError):
        return None


def cur_str_of(path):
    """<cur_str> of an archived <ts>_make_<cur_str>[_job<ID>...].bash; the bare stem if the
    name doesn't follow that pattern."""
    stem = os.path.splitext(os.path.basename(path))[0]
    m = _ARCHIVE_NAME.match(stem)
    return m.group('cur_str') if m else stem


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--user', default=getpass.getuser())
    ap.add_argument('--archive', default=None,
                    help=f'archive dir (default: {DEFAULT_ARCHIVE}, else {FALLBACK_ARCHIVE})')
    args = ap.parse_args()

    archive_dir = args.archive
    if archive_dir is None:
        archive_dir = DEFAULT_ARCHIVE if os.path.isdir(DEFAULT_ARCHIVE) else FALLBACK_ARCHIVE
    if not os.path.isdir(archive_dir):
        sys.exit(f'ERROR: archive directory not found: {archive_dir}')

    jobs = queued_jobs(args.user)
    if not jobs:
        print(f'No queued or running jobs for user {args.user}')
        return

    bash_files = sorted(f for f in os.listdir(archive_dir) if f.endswith('.bash'))

    paths = []        # matched files, de-duplicated (a resubmitted file can carry several IDs)
    unmatched = []
    cur_strs = {}     # job ID -> cur_str(s) of its archived file(s), '-' if none
    totals = {}       # job ID -> tests its archived generator made (None: unknown)
    for job_id in jobs:
        matches = archived_scripts_for(job_id, archive_dir, bash_files)
        if not matches:
            unmatched.append(job_id)
        cur_strs[job_id] = ','.join(dict.fromkeys(cur_str_of(p) for p in matches)) or '-'
        # several files for one ID shouldn't happen; then the total is ambiguous
        totals[job_id] = total_tests(matches[0]) if len(matches) == 1 else None
        for path in matches:
            if path not in paths:
                paths.append(path)

    for path in paths:
        print(f'vi {path}')

    print()
    width = max(len(c) for c in cur_strs.values())
    for job_id, states in jobs.items():
        running = states.get('RUNNING', 0)
        pending = states.get('PENDING', 0)
        other = sum(states.values()) - running - pending
        total = totals[job_id]
        line = (f'{job_id}  {cur_strs[job_id]:<{width}}  '
                f'{running:4d} running out of {total if total is not None else "?":>4} total tests'
                f'  (pending={pending}')
        if total is not None:
            # finished tasks (done or failed) leave the queue
            line += f', left the queue={max(total - sum(states.values()), 0)}'
        line += ')'
        if other:
            line += '  other=' + ','.join(f'{s}:{n}' for s, n in states.items()
                                          if s not in ('RUNNING', 'PENDING'))
        print(line)

    if unmatched:
        print(f'\nNo archived .bash file for job(s): {" ".join(unmatched)}'
              f'  (submitted without submit_and_record.bash / record_job_id.bash?)')


if __name__ == '__main__':
    main()
