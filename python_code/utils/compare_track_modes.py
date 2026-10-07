"""compare_track_modes.py -- overlay several ekf.py tracking modes (weights_track_mode, the t=<mode>
filename token) on one BLER / MI / BER figure, for one experiment tag.

Usage (from C:\\Projects\\DeepOFDM):
    python python_code\\utils\\compare_track_modes.py trkkfff --modes ekfcrc ekfbp ekfh --filter R=0.48
    python python_code\\utils\\compare_track_modes.py trkkfff --modes ekfcrc ekfh --filter R=0.37 cfo=0.20 --seeds 123
    python python_code\\utils\\compare_track_modes.py drrft --modes ekfbp sgdsbp --filter R=0.48 bpi=10 --out C:\\tmp\\x.png

Data sources (both are read; the newest timestamp wins per (config, seed, SNR)):
  - loose result CSVs in C:\\Projects\\Scratchpad:  <ts>_<config>_<tag>_s=<seed>_SNR=<snr>[_bler|_mi].csv
  - the archive C:\\Projects\\Scratchpad\\<tag>.gz (tar.gz of the same CSVs), if it exists
    (or any archive given with --archive).

Per (seed, SNR) the plotted value is the mean over the CSV's rows (groups) - the same as
plot_multiple_csvs._col_mean. Several seeds are averaged per SNR (seed counts shown in the
legend when they differ across SNRs). Missing SNRs are left as gaps (no interpolation). The
LMMSE reference (dashed gray) comes from the first mode. Each mode must resolve to exactly one
configuration after --filter; otherwise the script lists the candidates and stops, so you can add
a disambiguating --filter token (e.g. bpi=10).

--filter tokens are exact '_'-separated filename tokens (e.g. R=0.48, cfo=0.20, bpi=10, lr=2e-2).
Output: Analysis\\<tag>_<filters>\\compare_<modes>.png unless --out is given; the per-SNR table is
printed and, with --table, also written as CSV next to the figure.
"""
import argparse, collections, glob, io, os, re, sys, tarfile

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRATCH = r"C:\Projects\Scratchpad"
ANALYSIS = os.path.join(SCRATCH, "Analysis")
LONG = lambda p: ("\\\\?\\" + os.path.abspath(p)) if os.name == "nt" else p   # MAX_PATH-safe

# Fixed categorical order (validated palette: blue, orange, aqua, yellow, magenta, green, violet, red);
# markers are a second, non-color identity channel.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
LMMSE_COLOR = "#8a8a85"
METRICS = [("BLER", "_bler", "log"), ("MI", "_mi", "linear"), ("BER", "", "log")]
NAME_RE = re.compile(r"^(\d{8}_\d{4})_(.*)_s=(\d+)_SNR=(-?\d+)(_bler|_mi)?\.csv$")


def _record(store, name, read_fn, tag):
    """Parse one CSV name; if it belongs to tag, read it and keep it if newest for its slot."""
    m = NAME_RE.match(name)
    if not m:
        return
    ts, config, seed, snr, kind = m.groups()
    toks = config.split("_")
    if tag not in toks:
        return
    key = (config, kind or "", int(seed), int(snr))
    if key in store and store[key][0] >= ts:
        return
    df = pd.read_csv(read_fn())
    store[key] = (ts, float(df["total_ber_1"].mean()), float(df["total_ber_lmmse"].mean()))


def load(tag, archive=None):
    """-> {(config, kind, seed, snr): (ts, escnn_mean, lmmse_mean)}"""
    store = {}
    for p in glob.glob(os.path.join(SCRATCH, f"*_{tag}_s=*_SNR=*.csv")):
        _record(store, os.path.basename(p), lambda p=p: LONG(p), tag)
    archive = archive or os.path.join(SCRATCH, f"{tag}.gz")
    if os.path.isfile(archive):
        with tarfile.open(archive) as tf:
            for mem in tf.getmembers():
                if mem.isfile() and mem.name.endswith(".csv"):
                    _record(store, mem.name.split("/")[-1],
                            lambda mem=mem: io.BytesIO(tf.extractfile(mem).read()), tag)
    return store


def mode_of(config):
    return next((t[2:] for t in config.split("_") if t.startswith("t=")), None)


def crossing(snrs, vals, target=0.1):
    """First SNR where vals drops below target (interpolated in log10(vals))."""
    for i in range(1, len(snrs)):
        a, b = vals[i - 1], vals[i]
        if a >= target > b:
            la, lb = np.log10(max(a, 1e-9)), np.log10(max(b, 1e-9))
            return snrs[i - 1] + (np.log10(target) - la) / (lb - la) * (snrs[i] - snrs[i - 1])
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tag")
    ap.add_argument("--modes", nargs="+", required=True, help="t=<mode> values, in legend/color order")
    ap.add_argument("--filter", nargs="*", default=[], help="exact filename tokens every config must contain")
    ap.add_argument("--seeds", nargs="*", type=int, default=None, help="restrict to these seeds (default: all)")
    ap.add_argument("--archive", default=None, help="tar.gz to read besides loose CSVs (default Scratchpad/<tag>.gz)")
    ap.add_argument("--out", default=None, help="output .png path")
    ap.add_argument("--table", action="store_true", help="also write the per-SNR table as CSV")
    args = ap.parse_args()
    if len(args.modes) > len(COLORS):
        sys.exit(f"at most {len(COLORS)} modes per figure (fixed categorical palette) - split into two figures")

    store = load(args.tag, args.archive)
    if not store:
        sys.exit(f"no CSVs for tag {args.tag!r} in {SCRATCH} or its archive")
    flt = set(args.filter)

    # config per mode
    configs = collections.defaultdict(set)
    for (config, kind, seed, snr) in store:
        if flt <= set(config.split("_")) and (args.seeds is None or seed in args.seeds):
            configs[mode_of(config)].add(config)
    chosen = {}
    for mode in args.modes:
        c = sorted(configs.get(mode, ()))
        if len(c) != 1:
            print(f"mode {mode!r}: {len(c)} matching configs" + (" - add a --filter token to pick one:" if c else ""))
            for x in c:
                print("   ", x)
            sys.exit(1)
        chosen[mode] = c[0]

    # aggregate: data[mode][metric][snr] -> (escnn, lmmse, n_seeds)
    data = {m: {met: {} for met, _, _ in METRICS} for m in args.modes}
    for mode, config in chosen.items():
        for met, kind, _ in METRICS:
            per_snr = collections.defaultdict(list)
            for (c, k, seed, snr), (ts, e, l) in store.items():
                if c == config and k == kind and (args.seeds is None or seed in args.seeds):
                    per_snr[snr].append((e, l))
            for snr, v in per_snr.items():
                data[mode][met][snr] = (np.mean([x[0] for x in v]), np.mean([x[1] for x in v]), len(v))

    grid = sorted({s for m in data.values() for d in m.values() for s in d})
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4))
    for ax, (met, _, scale) in zip(axes, METRICS):
        ref = data[args.modes[0]][met]
        ax.plot(grid, [ref[s][1] if s in ref else np.nan for s in grid],
                color=LMMSE_COLOR, ls="--", lw=1.5, label="LMMSE")
        for i, mode in enumerate(args.modes):
            d = data[mode][met]
            snrs = sorted(d)
            lab = mode
            if met == "BLER":
                x = crossing(snrs, [d[s][0] for s in snrs])
                lab += f"  (10% @ {x:.1f} dB)" if x is not None else "  (10% not reached)"
            seeds = sorted({d[s][2] for s in snrs})
            if len(snrs) < len(grid):
                lab += f"  [{len(snrs)}/{len(grid)} SNRs]"
            if seeds and (len(seeds) > 1 or seeds[0] > 1):
                lab += f"  [{'-'.join(map(str, (seeds[0], seeds[-1]))) if len(seeds) > 1 else seeds[0]} seeds]"
            ax.plot(grid, [d[s][0] if s in d else np.nan for s in grid], color=COLORS[i],
                    marker=MARKERS[i], ms=6, lw=2, label=lab)
        if scale == "log":
            ax.set_yscale("log")
        else:
            ax.set_ylim(0, 1)
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel(f"{met} (ESCNN)")
        ax.grid(True, which="major", color="#e4e4e0", lw=0.8)
        ax.grid(True, which="minor", color="#f1f1ee", lw=0.5)
        ax.legend(fontsize=9, frameon=False, loc="lower left")
    seed_txt = "all seeds" if args.seeds is None else "seeds " + ",".join(map(str, args.seeds))
    fig.suptitle(f"{args.tag}  {' '.join(args.filter)}  ({seed_txt})", fontsize=12)
    fig.tight_layout()

    out = args.out or os.path.join(ANALYSIS, "_".join([args.tag] + args.filter),
                                   "compare_" + "_".join(args.modes) + ".png")
    os.makedirs(LONG(os.path.dirname(out)), exist_ok=True)
    fig.savefig(LONG(out), dpi=130)
    print("saved", out)
    for mode, config in chosen.items():
        print(f"  {mode:8s} <- {config}")

    rows = []
    for s in grid:
        row = {"SNR": s, "LMMSE_BLER": data[args.modes[0]]["BLER"].get(s, (np.nan,) * 3)[1]}
        for mode in args.modes:
            for met, _, _ in METRICS:
                row[f"{mode}_{met}"] = data[mode][met].get(s, (np.nan,) * 3)[0]
        rows.append(row)
    table = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(table.round(3).to_string(index=False))
    if args.table:
        tpath = os.path.splitext(out)[0] + ".csv"
        table.to_csv(LONG(tpath), index=False)
        print("table", tpath)


if __name__ == "__main__":
    main()
