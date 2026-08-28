"""Parse an ekf.py --drift streaming log and plot BER/MI/BLER vs. CFO.

Run as a module so the python_code package imports resolve:
    python -m python_code.utils.plot_drift_log <log_file> [--output out.png]

Expects the per-group lines ekf.py's run_streaming_drift() prints, e.g.:
    [drift] 1428 groups x 1 slot(s)/group, starting at channel_drift_base_index=0,
        cfo=0.0 (drift=0.75 scs/sec), SNR=12dB, mcs=2
    [drift] group 0/1428 slots=0-0 ber_escnn=9.3006e-02 ber_lmmse=9.7842e-02
        bler_escnn=0.0000e+00 bler_lmmse=0.0000e+00 mi_escnn=0.7232 mi_lmmse=0.7057
        sinr_db(mean/min/max over REs+users)=2.9/-12.2/10.2

CFO per group is reconstructed with the exact formula ekf.py uses to advance it
(see ekf.py:259-261): elapsed_slots = group_idx * group_size_slots, then
cfo = base_cfo + cfo_drift * elapsed_slots * SLOT_LENGTH_SEC. A log with no CFO
drift (cfo_drift=0, the header's "(drift=... scs/sec)" suffix omitted) still
works - every group just lands at the same base cfo.
"""
import argparse
import io
import os
import platform
import re
import subprocess
import tempfile
import textwrap

import matplotlib.pyplot as plt

SCRATCHPAD_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "Scratchpad"))

try:
    from python_code.utils.constants import SLOT_LENGTH_SEC
except ImportError:
    SLOT_LENGTH_SEC = 0.5e-3  # 5G 30kHz-SCS slot duration (14 OFDM symbols) - keep in sync with constants.py

try:
    from python_code.coding.mcs_table import get_mcs
except ImportError:
    # Minimal standalone fallback (mirrors python_code/coding/mcs_table.py's mcs_data) so this
    # script still works when run outside the python_code package, without needing pandas.
    _MCS_TABLE = {
        0: (2, 0.1172), 1: (2, 0.1885), 2: (2, 0.3008), 3: (2, 0.4385), 4: (2, 0.5879),
        5: (4, 0.3692), 6: (4, 0.4238), 7: (4, 0.4785), 8: (4, 0.5401), 9: (4, 0.6016), 10: (4, 0.6426),
        11: (6, 0.4551), 12: (6, 0.5049), 13: (6, 0.5537), 14: (6, 0.6016), 15: (6, 0.6504),
        16: (6, 0.7022), 17: (6, 0.7539), 18: (6, 0.8027), 19: (6, 0.8525),
        20: (8, 0.6665), 21: (8, 0.6943), 22: (8, 0.7363), 23: (8, 0.7783), 24: (8, 0.8213),
        25: (8, 0.8643), 26: (8, 0.8950), 27: (8, 0.9258),
        28: (4, 0.8213), 29: (4, 0.8950), 30: (4, 0.9258), 31: (6, 0.8950),
        32: (6, 0.3), 33: (6, 0.35), 34: (6, 0.4), 35: (4, 0.7), 36: (2, 0.72), 37: (2, 0.82), 38: (2, 0.92),
    }

    def get_mcs(index):
        if index not in _MCS_TABLE:
            raise ValueError(f"MCS index {index} not found (valid: 0-38)")
        return _MCS_TABLE[index]

HEADER_RE = re.compile(
    r'\[drift\]\s+(?P<num_groups>\d+)\s+groups x (?P<group_size>\d+) slot\(s\)/group, '
    r'starting at channel_drift_base_index=(?P<base_index>\d+), '
    r'cfo=(?P<base_cfo>[-+\d.eE]+)(?:\s+\(drift=(?P<cfo_drift>[-+\d.eE]+)\s+scs/sec\))?, '
    r'SNR=(?P<snr>[-+\d.eE]+)dB, mcs=(?P<mcs>\d+)'
)

# freeze/speed for the title come from here instead of the header line above: ekf.py's own
# "[CSV] wrote <path>" line (_build_ekf_filename_suffix) already bakes both into the filename
# as "..._sp=<speed>_..._frz=<code>_...", so parsing it out means every existing log already
# has this - no need for ekf.py to print it twice. Tokens are found by underscore-splitting the
# basename rather than matching the whole filename's fixed order, so this survives the field
# order/set in _build_ekf_filename_suffix changing.
CSV_WROTE_RE = re.compile(r'\[CSV\]\s+wrote\s+(?P<path>\S+\.csv)')

WEIGHTS_RE = re.compile(r'\[drift\]\s+loaded pretrained weights:.*?#REs=(?P<num_res>\d+)')

# Matches both the tracked path's "(epoch N)" and the static-diagnostic path's "(static)" -
# see ekf_predict_update()/_log_static_syndrome_stats() in escnn_trainer.py. mean_p is optional
# so logs from before it was added (only mean_hard_sat) still parse.
HARD_SAT_RE = re.compile(
    r'\[ekf\]\s+user=\d+\s+it=\d+\s+slot=\d+/\d+\s+\([^)]*\)\s+'
    r'checks=\d+\s+mean_hard_sat=(?P<mean_hard_sat>[-+\d.eE]+)'
    r'(?:\s+mean_p=(?P<mean_p>[-+\d.eE]+))?'
)

GROUP_RE = re.compile(
    r'\[drift\]\s+group\s+(?P<group_idx>\d+)/\d+\s+slots=[\d\-]+\s+'
    r'ber_escnn=(?P<ber_escnn>[-+\d.eE]+)\s+ber_lmmse=(?P<ber_lmmse>[-+\d.eE]+)\s+'
    r'bler_escnn=(?P<bler_escnn>[-+\d.eE]+)\s+bler_lmmse=(?P<bler_lmmse>[-+\d.eE]+)\s+'
    r'mi_escnn=(?P<mi_escnn>[-+\d.eE]+)\s+mi_lmmse=(?P<mi_lmmse>[-+\d.eE]+)'
)


def parse_drift_log(path: str) -> dict:
    """Returns dict of parallel lists: cfo, ber_escnn, ber_lmmse, bler_escnn,
    bler_lmmse, mi_escnn, mi_lmmse, mean_hard_sat - one entry per parsed group
    line - plus a 'meta' key holding the run parameters from the log's header line (SNR,
    mcs, base_cfo, cfo_drift, group_size, base_index, num_groups) and, if a
    "[drift] loaded pretrained weights: ..." line is present, num_res (parsed
    from that checkpoint filename's "#REs=" token - the header line itself
    doesn't carry the RE count), for use in the plot title.

    Stateful single pass: tracks the most recent header so a log containing
    several concatenated drift runs is handled correctly, each run's groups
    reconstructed against its own header. 'meta' is the FIRST header seen
    (the common case is one run per log); later headers still update the CFO
    reconstruction correctly, they just aren't reflected in the title.
    """
    data = {k: [] for k in ('cfo', 'slots', 'ber_escnn', 'ber_lmmse', 'bler_escnn',
                             'bler_lmmse', 'mi_escnn', 'mi_lmmse', 'mean_hard_sat', 'mean_p')}
    meta = None
    num_res = None  # from the weights-loaded line, which ekf.py prints BEFORE the "N groups x ..."
                     # header - so this has to be tracked independently of meta, not nested under it

    # mean_hard_sat/mean_p are Kalman-tracking diagnostics (see escnn_trainer.py's
    # ekf_predict_update()) printed on their own "[ekf] ..." line(s) BEFORE the "[drift] group
    # ..." line for the slot(s) they belong to - possibly several lines per group (multiple
    # users/iterations, or group_size_slots > 1), so they're accumulated here and averaged into
    # that group's single value once the group line is reached. float('nan') when a group has
    # none (e.g. an older log predating these diagnostics, or log_train_every_epochs disabled;
    # mean_p specifically is also absent in logs from before it was added alongside
    # mean_hard_sat) so the plotted line simply leaves a gap there instead of lying about a value.
    pending_hard_sats = []
    pending_mean_ps = []

    group_size = base_cfo = cfo_drift = None
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            hard_sat_m = HARD_SAT_RE.search(line)
            if hard_sat_m:
                pending_hard_sats.append(float(hard_sat_m.group('mean_hard_sat')))
                if hard_sat_m.group('mean_p') is not None:
                    pending_mean_ps.append(float(hard_sat_m.group('mean_p')))
                continue

            header_m = HEADER_RE.search(line)
            if header_m:
                group_size = int(header_m.group('group_size'))
                base_cfo = float(header_m.group('base_cfo'))
                cfo_drift = float(header_m.group('cfo_drift') or 0.0)
                if meta is None:
                    meta = {
                        'num_groups': int(header_m.group('num_groups')),
                        'group_size': group_size,
                        'base_index': int(header_m.group('base_index')),
                        'base_cfo': base_cfo,
                        'cfo_drift': cfo_drift,
                        'snr': float(header_m.group('snr')),
                        'mcs': int(header_m.group('mcs')),
                    }
                    if num_res is not None:
                        meta['num_res'] = num_res
                continue

            weights_m = WEIGHTS_RE.search(line)
            if weights_m:
                num_res = int(weights_m.group('num_res'))
                if meta is not None:
                    meta.setdefault('num_res', num_res)
                continue

            csv_m = CSV_WROTE_RE.search(line)
            if csv_m:
                for tok in os.path.basename(csv_m.group('path')).split('_'):
                    if tok.startswith('sp=') and meta is not None and 'speed' not in meta:
                        try:
                            meta['speed'] = float(tok[len('sp='):])
                        except ValueError:
                            pass
                    elif tok.startswith('frz=') and meta is not None:
                        meta.setdefault('freeze', tok[len('frz='):])
                continue

            group_m = GROUP_RE.search(line)
            if not group_m:
                continue
            if group_size is None:
                raise ValueError(
                    f"{path}: found a group line before any '[drift] N groups x ...' "
                    f"header line, so CFO can't be reconstructed")

            group_idx = int(group_m.group('group_idx'))
            elapsed_slots = group_idx * group_size
            cfo = base_cfo + cfo_drift * elapsed_slots * SLOT_LENGTH_SEC

            data['cfo'].append(cfo)
            data['slots'].append(elapsed_slots)
            for key in ('ber_escnn', 'ber_lmmse', 'bler_escnn', 'bler_lmmse', 'mi_escnn', 'mi_lmmse'):
                data[key].append(float(group_m.group(key)))

            data['mean_hard_sat'].append(_mean(pending_hard_sats) if pending_hard_sats else float('nan'))
            data['mean_p'].append(_mean(pending_mean_ps) if pending_mean_ps else float('nan'))
            pending_hard_sats = []
            pending_mean_ps = []

    if not data['cfo']:
        raise ValueError(f"{path}: no '[drift] group ...' lines found")
    data['meta'] = meta or {}
    return data


def _format_title(meta: dict, user_title: str = None, wrap_width: int = 110) -> str:
    """Builds a (possibly multi-line) title: the run's parameters plus, last, an
    optional user-supplied label (e.g. the log's name) - all wrapped together as
    one paragraph to wrap_width characters, so the label lands on the same line
    as the trailing params when it fits rather than always getting its own row."""
    parts = []
    if 'snr' in meta:
        parts.append(f"SNR={meta['snr']:g}dB")
    if 'mcs' in meta:
        try:
            qm, code_rate = get_mcs(meta['mcs'])
            mod_data = 2 ** qm
            mod_text = {2: 'BPSK', 4: 'QPSK'}.get(mod_data, f'{mod_data}Q')
            parts.append(f"{mod_text}, R={code_rate:.2f}")
        except ValueError:
            parts.append(f"mcs={meta['mcs']}")
    if 'num_res' in meta:
        parts.append(f"REs={meta['num_res']}")
    if 'freeze' in meta:
        parts.append(f"freeze={meta['freeze']}")
    if 'speed' in meta:
        parts.append(f"speed={meta['speed']:g}m/s")
    if 'base_cfo' in meta:
        parts.append(f"cfo0={meta['base_cfo']:g}")
    if 'cfo_drift' in meta:
        parts.append(f"cfo_drift={meta['cfo_drift']:g} scs/sec")
    if 'group_size' in meta:
        parts.append(f"slots/group={meta['group_size']}")
    if 'num_groups' in meta:
        parts.append(f"num_groups={meta['num_groups']}")
    if user_title:
        parts.append(user_title)

    lines = textwrap.wrap(', '.join(parts), width=wrap_width) if parts else []
    return '\n'.join(lines)


def _mean(values) -> float:
    """Mean of values, ignoring NaN entries (mean_p can have gaps - see
    parse_drift_log). Returns NaN if every entry is NaN/the list is empty."""
    clean = [v for v in values if v == v]  # v == v is False only for NaN
    return sum(clean) / len(clean) if clean else float('nan')


def _padded_ylim(values, lo=-1.0, hi=1.0, pad_frac=0.1, min_span=0.05):
    """Y-limits zoomed to values' actual (non-NaN) range plus a bit of padding, clamped to
    [lo, hi] (mean_p's theoretical range) so the padding never overshoots past what's possible.
    Falls back to the full [lo, hi] range when there's no usable data (all-NaN, e.g. an older
    log predating mean_p), and enforces min_span so a near-constant run doesn't collapse to an
    illegibly flat line."""
    clean = [v for v in values if v == v]  # v == v is False only for NaN
    if not clean:
        return lo, hi
    vmin, vmax = min(clean), max(clean)
    if vmax - vmin < min_span:
        center = (vmax + vmin) / 2
        vmin, vmax = center - min_span / 2, center + min_span / 2
    pad = (vmax - vmin) * pad_frac
    return max(lo, vmin - pad), min(hi, vmax + pad)


def plot_drift_log(data: dict, title: str = None):
    # With no CFO drift, every group reconstructs to the exact same cfo (see parse_drift_log) -
    # plotting vs. CFO then collapses all points onto one x value instead of showing a trend.
    # Fall back to slot index, which is always distinct per group, in that case.
    use_slots = data.get('meta', {}).get('cfo_drift', 0.0) == 0.0
    xkey = 'slots' if use_slots else 'cfo'
    order = sorted(range(len(data[xkey])), key=lambda i: data[xkey][i])
    cfo = [data[xkey][i] * SLOT_LENGTH_SEC for i in order] if use_slots else [data[xkey][i] for i in order]
    xlabel = 'Time (s)' if use_slots else 'CFO (scs)'

    full_title = _format_title(data.get('meta', {}), title)
    title_lines = full_title.count('\n') + 1 if full_title else 0

    # Landscape: the figure as a whole is wider than tall, but the 4 subplots
    # are still stacked vertically (sharing the CFO x-axis) rather than side by
    # side. Figure height grows a bit with the title so a multi-line parameter
    # header doesn't eat into subplot space, and a bit more per extra subplot.
    # Top-to-bottom: BER, MI, BLER, mean_p.
    fig, (ax_ber, ax_mi, ax_bler, ax_meanp) = plt.subplots(
        4, 1, figsize=(12, 9 + 0.18 * title_lines), sharex=True)

    ax_ber.semilogy(cfo, [data['ber_lmmse'][i] for i in order],
                     label=f"LMMSE: mean BER={_mean(data['ber_lmmse']):.2f}", color='r')
    ax_ber.semilogy(cfo, [data['ber_escnn'][i] for i in order],
                     label=f"ESCNN: mean BER={_mean(data['ber_escnn']):.2f}", color='g')
    ax_ber.set_ylabel('BER')
    ax_ber.legend()
    ax_ber.grid(True, which='both')

    ax_mi.plot(cfo, [data['mi_lmmse'][i] for i in order],
               label=f"LMMSE: mean MI={_mean(data['mi_lmmse']):.2f}", color='r')
    ax_mi.plot(cfo, [data['mi_escnn'][i] for i in order],
               label=f"ESCNN: mean MI={_mean(data['mi_escnn']):.2f}", color='g')
    ax_mi.set_ylabel('MI')
    ax_mi.legend()
    ax_mi.grid(True)

    ax_bler.plot(cfo, [data['bler_lmmse'][i] for i in order],
                 label=f"LMMSE: mean BLER={_mean(data['bler_lmmse']):.2f}", color='r')
    ax_bler.plot(cfo, [data['bler_escnn'][i] for i in order],
                 label=f"ESCNN: mean BLER={_mean(data['bler_escnn']):.2f}", color='g')
    ax_bler.set_ylabel('BLER')
    ax_bler.legend()
    ax_bler.grid(True)

    # mean_p is a single tracked-model diagnostic (not an LMMSE-vs-ESCNN comparison like the
    # other 3) - the raw (unthresholded) mean of the syndrome check-satisfaction values p_j,
    # range [-1, 1] (+1 = every check confidently satisfied, -1 = every check confidently
    # violated, 0 = no information). NaN gaps (groups with no "[ekf] ..." line, or an older log
    # predating this field) are left as gaps rather than interpolated over. Y-limits are zoomed
    # to this run's actual data range (with padding) rather than the full [-1, 1] scale, since
    # a run usually only occupies a narrow band and the full range makes it look flat.
    ax_meanp.plot(cfo, [data['mean_p'][i] for i in order],
                  label=f"mean syndrome={_mean(data['mean_p']):.2f}", color='b')
    ax_meanp.set_ylabel('syndrome')
    ax_meanp.set_ylim(*_padded_ylim(data['mean_p']))
    ax_meanp.set_xlabel(xlabel)
    ax_meanp.legend()
    ax_meanp.grid(True)

    if full_title:
        fig.suptitle(full_title)
        fig.tight_layout(rect=[0, 0, 1, 1 - 0.018 * title_lines])
    else:
        fig.tight_layout()

    # BER's log-scale tick labels (e.g. "3x10^-1") are wider than the other subplots'
    # linear ones ("0.6"), so matplotlib's per-axes auto-padding otherwise leaves the
    # "BER" ylabel sitting further left than the rest.
    fig.align_ylabels([ax_ber, ax_mi, ax_bler, ax_meanp])
    return fig


def _copy_fig_to_clipboard(fig):
    """Copy fig to the Windows clipboard as an image (CF_DIB), so it can be pasted directly
    into OneNote/Word/etc. Windows-only - needs pywin32 and Pillow.

    Rendered at 96 DPI (not the higher DPI a saved PNG would normally use) because paste
    targets like OneNote assume 96 DPI for pasted bitmaps and display them at
    pixel_size/96 inches - a higher DPI here would produce a huge on-page image."""
    if platform.system() != "Windows":
        raise RuntimeError("clipboard copy is Windows-only (uses win32clipboard)")
    import win32clipboard
    from PIL import Image

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=96, bbox_inches="tight")
    buf.seek(0)
    image = Image.open(buf).convert("RGB")

    bmp_buf = io.BytesIO()
    image.save(bmp_buf, "BMP")
    dib_data = bmp_buf.getvalue()[14:]  # strip the 14-byte BMP file header - CF_DIB wants the rest

    win32clipboard.OpenClipboard()
    try:
        win32clipboard.EmptyClipboard()
        win32clipboard.SetClipboardData(win32clipboard.CF_DIB, dib_data)
    finally:
        win32clipboard.CloseClipboard()


def _open_in_viewer(fig):
    """Save fig to a temp PNG and open it directly in mspaint.exe, a plain Win32 executable
    always present on Windows - so the plot stays on screen after this script exits instead of
    blocking on it the way plt.show() does (its window lives in this process and holds the
    terminal until closed).

    Deliberately subprocess.Popen(["mspaint.exe", ...]) rather than os.startfile() or
    webbrowser.open(): both of those route through ShellExecute/the OS's file-association
    handler for .png, which on this machine launches *something* without error but produces no
    visible window (confirmed: webbrowser.get() resolves to WindowsDefault, whose .open() is
    itself just os.startfile() - so that "fix" was the same silent-failure path in disguise).
    Launching a known executable directly sidesteps file-association entirely and raises a real,
    visible error (FileNotFoundError) if it can't start, instead of failing silently."""
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.close()
    # 96 DPI (Paint opens the canvas at 1 image px = 1 screen px, unlike apps that scale a
    # saved-PNG DPI tag) keeps the window a normal on-screen size instead of the ~1350x750px
    # a print-quality 150 DPI render would produce.
    fig.savefig(tmp.name, dpi=96, bbox_inches="tight")
    subprocess.Popen(["mspaint.exe", tmp.name])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('log_file', type=str, help='Path to the ekf.py --drift log to parse')
    parser.add_argument('--output', type=str, default=None,
                         help='Optional path to also save the plot (PNG, 150 DPI).')
    parser.add_argument('--title', type=str, default=None,
                         help='Label for the last title line (default: the log file\'s name).')
    args = parser.parse_args()

    log_file = args.log_file
    if not os.path.exists(log_file):
        # Bare filename (or a path that just doesn't exist here) - fall back to Scratchpad,
        # where ekf.py's sbatch runs actually land, rather than requiring an absolute path
        # every time.
        candidate = os.path.join(SCRATCHPAD_DIR, os.path.basename(log_file))
        if os.path.exists(candidate):
            log_file = candidate
        else:
            parser.error(f"log file not found: {args.log_file!r} (also checked {candidate!r})")

    title = args.title if args.title is not None else os.path.splitext(os.path.basename(log_file))[0]

    parsed = parse_drift_log(log_file)
    fig = plot_drift_log(parsed, title=title)

    if args.output:
        fig.savefig(args.output, dpi=150)
        print(f"[plot_drift_log] saved {args.output}")

    # Always copy to the clipboard (e.g. to paste into OneNote) - a failure here (non-Windows,
    # pywin32/Pillow missing) shouldn't stop the plot itself from showing.
    try:
        _copy_fig_to_clipboard(fig)
        print("[plot_drift_log] plot copied to clipboard")
    except Exception as e:
        print(f"[plot_drift_log] could not copy to clipboard: {e}")

    # Hand the image off to the OS viewer (its own process) instead of plt.show(), which would
    # block this terminal until the window is closed.
    try:
        _open_in_viewer(fig)
    except Exception as e:
        print(f"[plot_drift_log] could not open OS image viewer ({e}); falling back to plt.show()")
        plt.show()
