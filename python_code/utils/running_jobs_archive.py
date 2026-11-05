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
queued job ID with its cur_str (e.g. icpilot), running / pending / other task counts, how long
its longest-running task has been running (squeue %M) and, in the last columns, the augment(s),
MCS(s) and cfo(s) its generator swept, then any job IDs that have no archived file.
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
    """-> ({array-parent job ID (%F): Counter of task states},
           {job ID: seconds the longest-running of its RUNNING tasks has run (%M)}),
    for the user's queued jobs. A job with no running task has no entry in the second dict."""
    try:
        out = subprocess.run(['squeue', '-u', user, '-h', '-r', '-o', '%F %T %M'],
                             capture_output=True, text=True, check=True).stdout
    except FileNotFoundError:
        sys.exit('ERROR: squeue not found - run this on the Slurm login node')
    except subprocess.CalledProcessError as e:
        sys.exit(f'ERROR: squeue failed: {e.stderr.strip()}')
    jobs = defaultdict(Counter)
    oldest = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[0].isdigit():
            jobs[parts[0]][parts[1]] += 1
            if parts[1] == 'RUNNING':
                sec = _slurm_secs(parts[2])
                if sec is not None and sec > oldest.get(parts[0], -1):
                    oldest[parts[0]] = sec
    return dict(sorted(jobs.items(), key=lambda kv: int(kv[0]))), oldest


def _slurm_secs(t):
    """squeue %M ('M:SS', 'H:MM:SS' or 'D-HH:MM:SS') -> seconds; None if unparsable."""
    days, _, hms = t.rpartition('-')
    try:
        nums = [int(x) for x in hms.split(':')]
        d = int(days) if days else 0
    except ValueError:
        return None
    while len(nums) < 3:
        nums.insert(0, 0)
    h, m, s = nums[-3:]
    return ((d * 24 + h) * 60 + m) * 60 + s


def _fmt_dur(sec):
    """Seconds -> '05h12m' / '1d03h20m' (same style as job_progress.py)."""
    d, rem = divmod(int(sec), 86400)
    h, rem = divmod(rem, 3600)
    return f'{d}d{h:02d}h{rem // 60:02d}m' if d else f'{h:02d}h{rem // 60:02d}m'


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


# (label, array in the Parameters section) of the sweeps shown per job line
_SWEEPS = (('aug', 'which_augment_vals'), ('mcs', 'mcs_vals'), ('cfo', 'cfos'), ('chan', 'channel_model_vals'))

def swept_values(path):
    """{label: [values]} of the _SWEEPS arrays of an archived make_multiple_tests.bash, read
    the same way as total_tests: its Parameters section (everything before `total_count=0`) is
    run in bash and each array is echoed. An array the script doesn't define -> []. None if
    the file can't be evaluated."""
    try:
        with open(path) as f:
            head, sep, _ = f.read().partition('\ntotal_count=0')
        if not sep:
            return None
        script = head + '\n' + ''.join(f'echo "{label}=${{{arr}[*]}}"\n' for label, arr in _SWEEPS)
        out = subprocess.run(['bash', '-s', '--', 'x.yaml'], input=script, capture_output=True,
                             text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    vals = {label: [] for label, _ in _SWEEPS}
    for line in out.splitlines():
        label, eq, rest = line.partition('=')
        if eq and label in vals:
            vals[label] = rest.split()
    return vals


def _fmt_sweeps(vals):
    """{label: [values]} -> ['aug=LMMSE,DEEPRX', 'mcs=28', 'cfo=0.0,0.15'], one column each
    ('?' if unknown). Augment names are shortened like make_multiple_tests.bash's aug tag
    (AUGMENT_LMMSE -> LMMSE, NO_AUGMENT -> NOAUG)."""
    if vals is None:
        return [f'{label}=?' for label, _ in _SWEEPS]
    short = {'NO_AUGMENT': 'NOAUG'}
    parts = []
    for label, _ in _SWEEPS:
        v = vals[label]
        if label == 'aug':
            v = [short.get(x, x.replace('AUGMENT_', '')) for x in v]
        parts.append(f'{label}={",".join(v) or "-"}')
    return parts


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

    jobs, oldest = queued_jobs(args.user)
    if not jobs:
        print(f'No queued or running jobs for user {args.user}')
        return

    bash_files = sorted(f for f in os.listdir(archive_dir) if f.endswith('.bash'))

    paths = []        # matched files, de-duplicated (a resubmitted file can carry several IDs)
    unmatched = []
    cur_strs = {}     # job ID -> cur_str(s) of its archived file(s), '-' if none
    totals = {}       # job ID -> tests its archived generator made (None: unknown)
    sweeps = {}       # job ID -> ['aug=...', 'mcs=...', 'cfo=...'] of its archived generator
    for job_id in jobs:
        matches = archived_scripts_for(job_id, archive_dir, bash_files)
        if not matches:
            unmatched.append(job_id)
        cur_strs[job_id] = ','.join(dict.fromkeys(cur_str_of(p) for p in matches)) or '-'
        # several files for one ID shouldn't happen; then the total is ambiguous
        totals[job_id] = total_tests(matches[0]) if len(matches) == 1 else None
        sweeps[job_id] = (_fmt_sweeps(swept_values(matches[0])) if len(matches) == 1
                          else ['-'] if not matches else _fmt_sweeps(None))
        for path in matches:
            if path not in paths:
                paths.append(path)

    for path in paths:
        print(f'vi {path}')

    print()
    width = max(len(c) for c in cur_strs.values())
    heads, tails = [], []
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
        tail = ''
        if job_id in oldest:   # how long the longest-running of its running tasks has been running
            tail += f'oldest running: {_fmt_dur(oldest[job_id])}'
        if other:
            tail += ('  ' if tail else '') + 'other=' + ','.join(
                f'{s}:{n}' for s, n in states.items() if s not in ('RUNNING', 'PENDING'))
        heads.append(line)
        tails.append(tail)
    hw = max(len(h) for h in heads)   # pad the (...) part so the 'oldest running' column lines up
    tw = max(len(t) for t in tails)   # pad that column so the aug / mcs / cfo columns line up
    # width of each aug / mcs / cfo column (a job without an archived file has a single '-')
    sw = [max((len(sw_[i]) for sw_ in sweeps.values() if i < len(sw_)), default=0)
          for i in range(len(_SWEEPS))]
    for job_id, h, t in zip(jobs, heads, tails):
        cols = '  '.join(f'{c:<{sw[i]}}' for i, c in enumerate(sweeps[job_id]))
        print(f'{h:<{hw}}  {t:<{tw}}  {cols}'.rstrip())

    if unmatched:
        print(f'\nNo archived .bash file for job(s): {" ".join(unmatched)}'
              f'  (submitted without submit_and_record.bash / record_job_id.bash?)')


if __name__ == '__main__':
    main()
