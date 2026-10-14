"""compare_track_modes.py -- overlay several ekf.py tracking modes (weights_track_mode, the t=<mode>
filename token) on one BLER / MI / BER figure.

Usage (from C:\\Projects\\DeepOFDM):
    python python_code\\utils\\compare_track_modes.py --modes ekfcrc ekfbp ekfh --filter R=0.48
    python python_code\\utils\\compare_track_modes.py --modes ekfcrc ekfh --filter R=0.37 cfo=0.20 --seeds 123
    python python_code\\utils\\compare_track_modes.py --zipfile drrft --modes ekfbp sgdsbp --filter R=0.48 bpi=10
    python python_code\\utils\\compare_track_modes.py --modes ekfcrc ekfh --filter R=0.37 --snr 10 15

Data source (exactly one; the newest timestamp wins per (config, seed, SNR)):
  - default: the loose result CSVs in C:\\Projects\\Scratchpad
    (<ts>_<config>_s=<seed>_SNR=<snr>[_bler|_mi].csv); no .gz is read.
  - --zipfile NAME: only C:\\Projects\\Scratchpad\\NAME.gz (or a full path to a .gz); the loose CSVs
    are ignored.

Per (seed, SNR) the plotted value is the mean over the CSV's rows (groups) - the same as
plot_multiple_csvs._col_mean. Several seeds are averaged per SNR (seed counts shown in the
legend when they differ across SNRs). Missing SNRs are left as gaps (no interpolation). The
reference (dashed gray) comes from the first mode: the primary receiver ESCNN is augmented with
(aug=aDEEPSIC -> DeepSIC, aDEEPRX -> DeepRx, aSPHERE -> Sphere; LMMSE otherwise, or when that
detector's column is missing from the CSV). Each mode must resolve to exactly one
configuration after --filter; otherwise the script lists the candidates and stops, so you can add
a disambiguating --filter token (e.g. bpi=10).

--filter tokens are exact '_'-separated filename tokens (e.g. R=0.48, cfo=0.20, bpi=10, lr=2e-2).
Output: Analysis\\<NAME or 'scratch'>_<filters>\\compare_<modes>.png unless --out is given; with --table the
per-SNR table is also written as CSV next to the figure (it is not printed). The figure is also copied to
the clipboard and opened in Paint (plot_drift_log's helpers) unless --no-open is given.
"""
import argparse, collections, glob, io, os, re, sys, tarfile

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:   # run as a script from DeepOFDM root -> sys.path[0] is this utils dir
    from plot_drift_log import _copy_fig_to_clipboard, _open_in_viewer
except ImportError:   # run as python -m python_code.utils.compare_track_modes
    from python_code.utils.plot_drift_log import _copy_fig_to_clipboard, _open_in_viewer

SCRATCH = r"C:\Projects\Scratchpad"
ANALYSIS = os.path.join(SCRATCH, "Analysis")
LONG = lambda p: ("\\\\?\\" + os.path.abspath(p)) if os.name == "nt" else p   # MAX_PATH-safe

# Fixed categorical order (validated palette: blue, orange, aqua, yellow, magenta, green, violet, red);
# markers are a second, non-color identity channel.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
LMMSE_COLOR = "#8a8a85"
# aug=<token> -> (CSV column suffix, legend name) of the primary receiver used as the dashed reference
AUG_REF = {"aDEEPSIC": ("deepsic", "DeepSIC"), "aDEEPRX": ("deeprx", "DeepRx"), "aSPHERE": ("sphere", "Sphere")}
# Fixed palette slot per mode, so a mode looks the same in every figure regardless of --modes order.
# Modes not listed get a stable crc32-based slot (bumped to the next free slot only if it clashes
# with another mode in the same figure) - add them here to pin them permanently.
MODE_SLOT = {"ekfcrc": 0, "ekfbp": 1, "ekfh": 2, "ekfht": 3, "ekfi": 4, "sgdsbp": 5, "ekf": 6, "sgd": 7}
# (name, CSV kind, y scale, aggregation). "mean": mean over each CSV's rows (groups), then over seeds.
METRICS = [("BLER", "_bler", "log", "mean"), ("MI", "_mi", "linear", "mean"),
           ("BER", "", "log", "mean")]
NAME_RE = re.compile(r"^(\d{8}_\d{4})_(.*)_s=(\d+)_SNR=(-?\d+)(_bler|_mi)?\.csv$")


def _record(store, name, read_fn):
    """Parse one CSV name; read it and keep it if newest for its slot."""
    m = NAME_RE.match(name)
    if not m:
        return
    ts, config, seed, snr, kind = m.groups()
    key = (config, kind or "", int(seed), int(snr))
    if key in store and store[key][0] >= ts:
        return
    df = pd.read_csv(read_fn())
    aug = next((t[4:] for t in config.split("_") if t.startswith("aug=")), None)
    col, ref_name = AUG_REF.get(aug, ("lmmse", "LMMSE"))
    if f"total_ber_{col}" not in df:
        col, ref_name = "lmmse", "LMMSE"
    store[key] = (ts, df["total_ber_1"].to_numpy(float), df[f"total_ber_{col}"].to_numpy(float), ref_name)


def zip_path(name):
    """--zipfile value -> archive path: a bare name means Scratchpad\\<name>.gz."""
    if os.path.dirname(name) or name.endswith(".gz"):
        return name
    return os.path.join(SCRATCH, name + ".gz")


def load(zipfile=None):
    """-> {(config, kind, seed, snr): (ts, escnn_per_group, ref_per_group, ref_name)}
    From the given archive only, or (zipfile=None) from the loose Scratchpad CSVs only."""
    store = {}
    if zipfile is None:
        for p in glob.glob(os.path.join(SCRATCH, "*_s=*_SNR=*.csv")):
            _record(store, os.path.basename(p), lambda p=p: LONG(p))
        return store
    with tarfile.open(zipfile) as tf:
        for mem in tf.getmembers():
            if mem.isfile() and mem.name.endswith(".csv"):
                _record(store, mem.name.split("/")[-1],
                        lambda mem=mem: io.BytesIO(tf.extractfile(mem).read()))
    return store


def mode_slots(modes):
    """mode -> palette index: MODE_SLOT for known modes, else stable hash; unique within the figure."""
    import zlib
    slots = {m: MODE_SLOT[m] for m in modes if m in MODE_SLOT}
    for m in sorted(set(modes) - set(slots)):
        s = zlib.crc32(m.encode()) % len(COLORS)
        while s in slots.values():
            s = (s + 1) % len(COLORS)
        slots[m] = s
    return slots


def mode_of(config):
    return next((t[2:] for t in config.split("_") if t.startswith("t=")), None)


def crossing(snrs, vals, step, target=0.1):
    """First SNR where vals drops below target -> (lo, hi).
    If the bracketing SNRs are adjacent on the grid (spacing <= step), interpolate in log10(vals) and
    return (x, x). If a missing SNR lies between them, interpolating across the gap is meaningless on
    a steep waterfall (it made identical curves report different crossings), so return the bracket
    (snr_before, snr_after) instead."""
    for i in range(1, len(snrs)):
        a, b = vals[i - 1], vals[i]
        if a >= target > b:
            lo, hi = snrs[i - 1], snrs[i]
            if hi - lo > step:
                return lo, hi
            la, lb = np.log10(max(a, 1e-9)), np.log10(max(b, 1e-9))
            x = lo + (np.log10(target) - la) / (lb - la) * (hi - lo)
            return x, x
    return None


def bler10_label(snrs, vals, step):
    """Legend suffix giving where a BLER curve crosses 10%."""
    x = crossing(snrs, vals, step)
    if x is not None and x[0] == x[1]:
        return f"  (10% @ {x[0]:.1f} dB)"
    if x is not None:
        return f"  (10% in {x[0]:g}-{x[1]:g} dB, gap)"
    if vals and vals[0] < 0.1:   # already below 10% at its first available SNR
        return f"  (10% @ <= {snrs[0]} dB)"
    return "  (10% not reached)"


def make_figure(store, chosen, modes, title, out, seeds=None, table=False, snr_range=None):
    """Draw the BLER/MI/BER comparison of `modes` (legend labels, in order; chosen[label] is the
    config key each one plots) from a load() store, save it to `out` and return the figure.
    Labels found in MODE_SLOT keep their fixed color. With table=True the per-SNR table is also
    written as CSV next to the figure. snr_range=(lo, hi) keeps only SNRs in [lo, hi] (inclusive).
    chosen[label] may also be a tuple of config keys, pooled together (e.g. one per seed when each
    seed's run carries its own w=/r= hash)."""
    cfgs = lambda v: (v,) if isinstance(v, str) else tuple(v)

    # aggregate: data[mode][metric][snr] -> (escnn, ref, n_seeds); ref = primary receiver (see AUG_REF)
    data = {m: {met: {} for met, *_ in METRICS} for m in modes}
    for mode, config in chosen.items():
        for met, kind, _, agg in METRICS:
            per_snr = collections.defaultdict(list)
            for (c, k, seed, snr), (ts, e, l, _) in store.items():
                if c in cfgs(config) and k == kind and (seeds is None or seed in seeds)                         and (snr_range is None or snr_range[0] <= snr <= snr_range[1]):
                    per_snr[snr].append((e, l))
            for snr, v in per_snr.items():
                ev, lv = np.mean([x[0].mean() for x in v]), np.mean([x[1].mean() for x in v])
                data[mode][met][snr] = (ev, lv, len(v))
    # code rate from the R=<r> token (same for all modes after --filter, else taken from the first)
    rate = next((float(t[2:]) for t in cfgs(chosen[modes[0]])[0].split("_") if t.startswith("R=")), None)

    # reference name from the first mode's CSVs (falls back to LMMSE per file - see _record)
    ref_names = {v[3] for (c, *_), v in store.items() if c in cfgs(chosen[modes[0]])}
    ref_name = ref_names.pop() if len(ref_names) == 1 else "LMMSE"

    grid = sorted({s for m in data.values() for d in m.values() for s in d})
    step = min(np.diff(grid)) if len(grid) > 1 else 0
    slot = mode_slots(modes)
    fig, axes = plt.subplots(1, len(METRICS), figsize=(6 * len(METRICS), 5.4))
    for ax, (met, _, scale, _) in zip(axes, METRICS):
        ref = data[modes[0]][met]
        lab = ref_name
        if met == "BLER":
            rs = sorted(ref)
            lab += bler10_label(rs, [ref[s][1] for s in rs], step)
        ax.plot(grid, [ref[s][1] if s in ref else np.nan for s in grid],
                color=LMMSE_COLOR, ls="--", lw=1.5, label=lab)
        for mode in modes:
            i = slot[mode]   # fixed per mode name - same color/marker in every figure
            d = data[mode][met]
            snrs = sorted(d)
            lab = mode
            if met == "BLER":
                lab += bler10_label(snrs, [d[s][0] for s in snrs], step)
            nseeds = sorted({d[s][2] for s in snrs})
            if len(snrs) < len(grid):
                lab += f"  [{len(snrs)}/{len(grid)} SNRs]"
            if nseeds and (len(nseeds) > 1 or nseeds[0] > 1):
                lab += f"  [{'-'.join(map(str, (nseeds[0], nseeds[-1]))) if len(nseeds) > 1 else nseeds[0]} seeds]"
            ax.plot(grid, [d[s][0] if s in d else np.nan for s in grid], color=COLORS[i],
                    marker=MARKERS[i], ms=6, lw=2, label=lab)
        if scale == "log":
            ax.set_yscale("log")
            if met == "BLER":   # always show at least down to 1e-2 (lower if the data goes lower)
                ax.set_ylim(bottom=min(ax.get_ylim()[0], 1e-2))
        else:
            ax.set_ylim(0, 1)
            if rate is not None:   # MI panels: a group needs MI above R to decode
                ax.axhline(rate, color="k", ls=":", lw=1.2, label=f"R={rate:g}")
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel(f"{met} (ESCNN)")
        ax.grid(True, which="major", color="#e4e4e0", lw=0.8)
        ax.grid(True, which="minor", color="#f1f1ee", lw=0.5)
        ax.legend(fontsize=9, frameon=False, loc="lower left")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()

    os.makedirs(LONG(os.path.dirname(out)), exist_ok=True)
    fig.savefig(LONG(out), dpi=130)
    print("saved", out)
    for mode, config in chosen.items():
        print(f"  {mode:8s} <- {' + '.join(cfgs(config))}")

    rows = []
    for s in grid:
        row = {"SNR": s, f"{ref_name}_BLER": data[modes[0]]["BLER"].get(s, (np.nan,) * 3)[1]}
        for mode in modes:
            for met, *_ in METRICS:
                row[f"{mode}_{met}"] = data[mode][met].get(s, (np.nan,) * 3)[0]
        rows.append(row)
    if table:
        tpath = os.path.splitext(out)[0] + ".csv"
        pd.DataFrame(rows).to_csv(LONG(tpath), index=False)
        print("table", tpath)
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zipfile", default=None,
                    help="read only Scratchpad/<NAME>.gz (or a .gz path) instead of the loose Scratchpad CSVs")
    ap.add_argument("--modes", nargs="+", required=True, help="t=<mode> values, in legend/color order")
    ap.add_argument("--filter", nargs="*", default=[], help="exact filename tokens every config must contain")
    ap.add_argument("--seeds", nargs="*", type=int, default=None, help="restrict to these seeds (default: all)")
    ap.add_argument("--snr", nargs=2, type=int, default=None, metavar=("LO", "HI"),
                    help="plot only SNRs in [LO, HI] dB (inclusive), e.g. --snr 10 15")
    ap.add_argument("--out", default=None, help="output .png path")
    ap.add_argument("--table", action="store_true", help="also write the per-SNR table as CSV")
    ap.add_argument("--no-open", dest="open", action="store_false",
                    help="do not copy the figure to the clipboard or open it in Paint")
    args = ap.parse_args()
    if len(args.modes) > len(COLORS):
        sys.exit(f"at most {len(COLORS)} modes per figure (fixed categorical palette) - split into two figures")

    if args.zipfile:
        zpath = zip_path(args.zipfile)
        if not os.path.isfile(zpath):
            sys.exit(f"archive not found: {zpath}")
        src = os.path.splitext(os.path.basename(zpath))[0]
        store = load(zpath)
        where = zpath
    else:
        src, where = "scratch", f"loose CSVs in {SCRATCH}"
        store = load()
    if not store:
        sys.exit(f"no result CSVs in {where}")
    print(f"reading {len(store)} CSVs from {where}")
    flt = set(args.filter)

    # config per mode
    configs = collections.defaultdict(set)
    for (config, kind, seed, snr) in store:
        if flt <= set(config.split("_")) and (args.seeds is None or seed in args.seeds):
            configs[mode_of(config)].add(config)
    chosen, missing = {}, []
    for mode in args.modes:
        c = sorted(configs.get(mode, ()))
        if not c:
            missing.append(mode)
            continue
        if len(c) > 1:
            print(f"mode {mode!r}: {len(c)} matching configs - add a --filter token to pick one:")
            for x in c:
                print("   ", x)
            sys.exit(1)
        chosen[mode] = c[0]
    if missing:
        print(f"WARNING: no logs found for mode(s) {', '.join(missing)} "
              f"(source {src!r}, filter {' '.join(args.filter) or '-'}) - plotting the rest")
    if not chosen:
        sys.exit("none of the requested modes has logs - nothing to plot")
    modes = [m for m in args.modes if m in chosen]   # keeps the requested order
    seed_txt = "all seeds" if args.seeds is None else "seeds " + ",".join(map(str, args.seeds))
    snr_txt = f"_snr{args.snr[0]}to{args.snr[1]}" if args.snr else ""
    out = args.out or os.path.join(ANALYSIS, "_".join([src] + args.filter),
                                   "compare_" + "_".join(modes) + snr_txt + ".png")
    fig = make_figure(store, chosen, modes, f"{src}  {' '.join(args.filter)}  ({seed_txt})", out,
                      seeds=args.seeds, table=args.table, snr_range=args.snr)
    if args.open:
        try:
            _copy_fig_to_clipboard(fig)
            print("plot copied to clipboard")
        except Exception as e:
            print(f"could not copy to clipboard: {e}")
        try:
            _open_in_viewer(fig)
        except Exception as e:
            print(f"could not open image viewer: {e}")


if __name__ == "__main__":
    main()
