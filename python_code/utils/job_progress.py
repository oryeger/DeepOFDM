#!/usr/bin/env python3
"""Show how many slots each running ekf.py job has completed out of its total, with an ETA.

Usage (on the cluster; logs are read from DeepOFDM/logs unless --logs-dir is given):
    ./job_progress.py                  # logs of the user's RUNNING array tasks (from squeue)
    ./job_progress.py --all            # every job-*.out in the logs directory
    ./job_progress.py 23421302               # all tasks of a job (looked up in the logs dir)
    ./job_progress.py 23421302_39 '*23433432*'   # a task, a quoted glob, or a path to a log

Per log it reads the lines ekf.py already prints:
    [drift] 89 groups x 1 slot(s)/group (...), track_mode=ekfbp, ..., SNR=8dB, mcs=10
    [drift] group 10/89 slots=10-10 ...          <- one line per completed group
    [CSV] wrote ...                              <- run finished
and the Slurm header date (start time). Elapsed time runs to the log's last modification;
the ETA assumes the remaining groups take as long as the completed ones did.

cur_str is taken from the archived generator script, via running_jobs_archive.py.

Note: Python block-buffers stdout when it goes to a file, so the log (and therefore the
count here) can lag behind the job's true progress by a few groups unless it runs with -u.
"""
import argparse
import datetime as dt
import getpass
import glob
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from running_jobs_archive import (DEFAULT_ARCHIVE, FALLBACK_ARCHIVE,  # noqa: E402
                                  archived_scripts_for, cur_str_of)

_HDR = re.compile(r'\[drift\] (\d+) groups x (\d+) slot\(s\)/group.*?track_mode=(\w+).*?SNR=(-?[\d.]+)dB(?:, mcs=(\d+))?')
_GROUP = re.compile(r'^\[drift\] group (\d+)/(\d+) ', re.M)
_START = re.compile(r'^(\w{3} \d{1,2} \w{3} \d{4} \d{2}:\d{2}:\d{2})', re.M)
_LOGNAME = re.compile(r'job-(\d+)(?:_(\d+))?\.out$')

# python_code/utils/ -> repo root -> logs (same resolution as running_jobs_archive's archive dir)
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DEFAULT_LOGS = os.path.join(_REPO_ROOT, 'logs')
FALLBACK_LOGS = '/home/egero/Projects/DeepOFDM/logs'


def running_tasks(user):
    """['<parent>_<task>' or '<jobid>'] of the user's RUNNING tasks (-r expands arrays)."""
    try:
        out = subprocess.run(['squeue', '-u', user, '-h', '-r', '-t', 'RUNNING', '-o', '%i'],
                             capture_output=True, text=True, check=True).stdout
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        sys.exit(f'ERROR: squeue failed ({e}) - run on the login node, or pass --all / files')
    return [t.strip() for t in out.split() if t.strip()]


def fmt_dur(sec):
    if sec is None:
        return '-'
    sec = int(sec)
    d, rem = divmod(sec, 86400)
    h, rem = divmod(rem, 3600)
    m = rem // 60
    return f'{d}d{h:02d}h{m:02d}m' if d else f'{h:02d}h{m:02d}m'


def parse_log(path):
    with open(path, errors='replace') as f:
        txt = f.read()
    r = {'file': os.path.basename(path), 'snr': '-', 'mode': '-', 'mcs': '-',
         'done': 0, 'total': None, 'spg': 1, 'finished': '[CSV] wrote' in txt,
         'elapsed': None, 'eta': None}
    m = _HDR.search(txt)
    if m:
        r['total'], r['spg'] = int(m.group(1)), int(m.group(2))
        r['mode'], r['snr'], r['mcs'] = m.group(3), m.group(4), m.group(5) or '-'
    groups = _GROUP.findall(txt)
    if groups:
        r['done'] = max(int(g) for g, _ in groups) + 1
        r['total'] = r['total'] or int(groups[-1][1])
    s = _START.search(txt)
    if s:
        try:
            start = dt.datetime.strptime(s.group(1), '%a %d %b %Y %H:%M:%S')
            end = dt.datetime.fromtimestamp(os.path.getmtime(path))
            r['elapsed'] = (end - start).total_seconds()
            if r['done'] and r['total'] and not r['finished']:
                r['eta'] = r['elapsed'] / r['done'] * (r['total'] - r['done'])
        except ValueError:
            pass
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('files', nargs='*',
                    help='log paths, job IDs (23421302 / 23421302_39) or globs, resolved in the '
                         'logs dir when not an existing path (default: logs of running tasks)')
    ap.add_argument('--logs-dir', default=None,
                    help=f'directory holding job-*.out (default: {DEFAULT_LOGS}, else {FALLBACK_LOGS})')
    ap.add_argument('--all', action='store_true', help='every job-*.out in --logs-dir')
    ap.add_argument('--user', default=getpass.getuser())
    args = ap.parse_args()
    if args.logs_dir is None:
        args.logs_dir = DEFAULT_LOGS if os.path.isdir(DEFAULT_LOGS) else FALLBACK_LOGS
    if not os.path.isdir(args.logs_dir) and not all(os.path.isfile(f) for f in args.files):
        sys.exit(f'ERROR: logs directory not found: {args.logs_dir}')

    if args.files:
        paths = []
        for a in args.files:
            if os.path.isfile(a):
                hits = [a]
            elif re.fullmatch(r'\d+(_\d+)?', a):          # job ID or job_task
                hits = sorted(glob.glob(os.path.join(args.logs_dir, f'job-{a}.out')) +
                              glob.glob(os.path.join(args.logs_dir, f'job-{a}_*.out')))
            else:                                            # glob / name, in the logs dir
                hits = sorted(glob.glob(a) or glob.glob(os.path.join(args.logs_dir, a)))
            if not hits:
                print(f'(no log matches {a!r} in {os.path.abspath(args.logs_dir)})')
            paths.extend(h for h in hits if h not in paths)
    elif args.all:
        paths = sorted(glob.glob(os.path.join(args.logs_dir, 'job-*.out')))
    else:
        paths = []
        for t in running_tasks(args.user):
            hits = glob.glob(os.path.join(args.logs_dir, f'job-{t}.out')) or \
                   glob.glob(os.path.join(args.logs_dir, f'job-{t}_*.out'))
            if hits:
                paths.extend(hits)
            else:
                print(f'(no log for running task {t} in {os.path.abspath(args.logs_dir)})')
    if not paths:
        sys.exit('No logs to parse.')

    archive = DEFAULT_ARCHIVE if os.path.isdir(DEFAULT_ARCHIVE) else FALLBACK_ARCHIVE
    bash_files = sorted(f for f in os.listdir(archive) if f.endswith('.bash')) \
        if os.path.isdir(archive) else []
    cur_cache = {}

    rows = []
    for p in paths:
        r = parse_log(p)
        m = _LOGNAME.search(os.path.basename(p))
        parent = m.group(1) if m else None
        if parent not in cur_cache:
            hits = archived_scripts_for(parent, archive, bash_files) if parent else []
            cur_cache[parent] = ','.join(dict.fromkeys(cur_str_of(h) for h in hits)) or '-'
        r['cur_str'] = cur_cache[parent]
        rows.append(r)

    hdr = ('log', 'cur_str', 'mode', 'mcs', 'SNR', 'slots done/total', '%', 'elapsed', 'ETA')
    table = []
    for r in rows:
        tot = r['total']
        done_s, tot_s = r['done'] * r['spg'], (tot * r['spg'] if tot else None)
        pct = f'{100 * r["done"] / tot:.0f}' if tot else '-'
        eta = 'done' if r['finished'] else fmt_dur(r['eta'])
        table.append((r['file'], r['cur_str'], r['mode'], r['mcs'], r['snr'],
                      f'{done_s}/{tot_s if tot_s else "?"}', pct, fmt_dur(r['elapsed']), eta))
    w = [max(len(str(x)) for x in col) for col in zip(hdr, *table)]
    print('  '.join(h.ljust(n) for h, n in zip(hdr, w)))
    for t in table:
        print('  '.join(str(x).ljust(n) for x, n in zip(t, w)))


if __name__ == '__main__':
    main()
