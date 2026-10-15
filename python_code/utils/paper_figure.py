"""
paper_figure.py -- one publication-ready BLER panel (IEEE single column, e.g. ICC) for one tag.

Usage:
    python paper_figure.py itrknk --seeds 123 --snr-max 26
    python paper_figure.py itrknk --seeds 123 --snr-max 26 --modes notrack ekfi ekfht --label ekfht="EKF-HT"

Data: the tag's loose CSVs in C:\\Projects\\Scratchpad, or - when none are left there - the tag's
archive Scratchpad\\<tag>.gz (same reader as compare_track_modes). Per (mode, SNR): mean BLER over
the CSV's rows, then over the requested seeds (each point lists how many seeds it has if they
differ). The primary receiver (LMMSE / DeepSIC / ...) is drawn dashed as the reference.
BLER = 0 points are left out (a log axis can't show them) and reported on the console.

Output: Analysis\\<tag>\\paper\\bler_<tag>_s=<seeds>_snr<lo>-<hi>.pdf (vector, fonts embedded as
TrueType - drop into Overleaf with \\includegraphics[width=\\columnwidth]{...}) and a 600-dpi .png.
Colors/markers per mode are compare_track_modes' fixed ones, so the figure matches the HTML pages;
markers keep the curves apart in grayscale print.
"""
import argparse, collections, glob, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import compare_track_modes as ctm

# Legend text per mode - edit here (or pass --label mode=Text) to match the paper's naming.
LABELS = {}


def load_store(tag):
    loose = glob.glob(os.path.join(ctm.SCRATCH, f"*_{tag}_s=*_SNR=*.csv"))
    if loose:
        store = {}
        for p in loose:
            ctm._record(store, os.path.basename(p), lambda p=p: ctm.LONG(p))
        return store, "loose CSVs"
    gz = os.path.join(ctm.SCRATCH, f"{tag}.gz")
    if not os.path.exists(gz):
        raise SystemExit(f"No CSVs or {gz} for tag '{tag}'")
    return ctm.load(gz), gz


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tag")
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--snr-min", type=float, default=-np.inf)
    ap.add_argument("--snr-max", type=float, default=np.inf)
    ap.add_argument("--modes", nargs="*", default=None, help="modes to draw, in legend order (default: all)")
    ap.add_argument("--filter", nargs="*", default=[], help="config tokens that must be present, e.g. R=0.48")
    ap.add_argument("--label", nargs="*", default=[], help="mode=Legend text overrides")
    ap.add_argument("--no-ref", action="store_true", help="don't draw the primary receiver reference")
    ap.add_argument("--width", type=float, default=3.5, help="inches (IEEE column = 3.5)")
    ap.add_argument("--height", type=float, default=2.6, help="inches")
    args = ap.parse_args()
    labels = dict(LABELS, **dict(l.split("=", 1) for l in args.label))

    store, src = load_store(args.tag)
    flt = set(args.filter)
    # (config, seed, snr) -> (escnn BLER, ref BLER, ref name), BLER CSVs of this tag only
    rows = {(c, sd, snr): (e.mean(), l.mean(), rn) for (c, k, sd, snr), (ts, e, l, rn) in store.items()
            if k == "_bler" and c.endswith("_" + args.tag) and sd in args.seeds
            and args.snr_min <= snr <= args.snr_max and flt <= set(c.split("_"))}
    configs = sorted({c for c, *_ in rows})
    by_mode = collections.defaultdict(set)
    for c in configs:
        by_mode[ctm.mode_of(c) or "?"].add(c)
    # everything except the tracking-mode family (and per-seed weight hashes) must agree
    strip = re.compile(r"_(t|trk|csg|lr|ep|bs|loss|tw|po|bpi|r|w)=[^_]+")
    bases = {strip.sub("", c) for c in configs}
    if len(bases) != 1:
        raise SystemExit("Several underlying configurations match - narrow with --filter:\n  " + "\n  ".join(sorted(bases)))
    modes = args.modes or sorted(by_mode, key=lambda m: (m != "notrack", ctm.MODE_SLOT.get(m, 99), m))
    missing = [m for m in modes if m not in by_mode]
    if missing:
        raise SystemExit(f"No data for mode(s) {missing}; available: {sorted(by_mode)}")

    def curve(m, idx):
        per_snr = collections.defaultdict(list)
        for (c, sd, snr), v in rows.items():
            if c in by_mode[m]:
                per_snr[snr].append(v[idx])
        return {s: (np.mean(v), len(v)) for s, v in sorted(per_snr.items())}

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 6.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(args.width, args.height))
    slot = ctm.mode_slots(modes)
    zeros, gaps = [], []
    grid = sorted({snr for (_, _, snr) in rows})   # every SNR any curve has
    # Bottom of the y-axis: half the lowest non-zero value drawn.
    _drawn = [curve(modes[0], 1)] if not args.no_ref else []
    _drawn += [curve(m, 0) for m in modes]
    _pos = [v[0] for d in _drawn for v in d.values() if v[0] > 0]
    Y_BOTTOM = 0.5 * min(_pos) if _pos else 1e-3
    FLOOR = 1e-12   # stand-in for BLER = 0: far below the axis, so the line drops (near-)vertically
                    # off the bottom edge and is clipped there - same look as the HTML pages

    def draw(d, name, **kw):
        # Missing SNRs become NaN, so the line breaks there instead of bridging the gap. BLER = 0
        # (no block errors): the first zero after a non-zero point drops the line straight down off the
        # bottom of the axis (the usual waterfall look); further zeros in the run are left empty.
        xs = grid
        ys, prev = [], np.nan
        for x in xs:
            if x not in d:
                y = np.nan
            elif d[x][0] > 0:
                y = d[x][0]
            else:
                y = FLOOR if (np.isfinite(prev) and prev > FLOOR) else np.nan
            ys.append(y)
            prev = y
        on_floor = [i for i, y in enumerate(ys) if y == FLOOR]
        if "marker" in kw and on_floor:   # no marker on the floor point - it isn't a measured value
            kw = dict(kw, markevery=[i for i in range(len(ys)) if i not in on_floor])
        zeros.extend((name, s) for s in d if d[s][0] <= 0)
        miss = [s for s in grid if min(d, default=0) < s < max(d, default=0) and s not in d]
        if miss:
            gaps.append((name, miss))
        counts = {d[s][1] for s in d}
        if len(counts) > 1:
            print(f"  [note] {name}: points average {min(counts)}-{max(counts)} seeds")
        ax.plot(xs, ys, label=name, **kw)

    if not args.no_ref:
        ref = curve(modes[0], 1)
        ref_name = {v[2] for (c, *_), v in rows.items() if c in by_mode[modes[0]]}
        draw(ref, ref_name.pop() if len(ref_name) == 1 else "LMMSE",
             color=ctm.LMMSE_COLOR, ls="--", lw=1.0)
    for m in modes:
        i = slot[m]
        draw(curve(m, 0), labels.get(m, m), color=ctm.COLORS[i], marker=ctm.MARKERS[i],
             ms=3.5, lw=1.1, markeredgewidth=0.6)

    ax.set_yscale("log")
    ax.set_xlabel("SNR [dB]")
    ax.set_ylabel("BLER")
    ax.grid(True, which="major", color="#d9d9d9", lw=0.5)
    ax.grid(True, which="minor", color="#efefef", lw=0.3)
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.set_xlim(min(s for (_, _, s) in rows), max(s for (_, _, s) in rows))
    ax.set_ylim(Y_BOTTOM, 1.1)
    # legend above the axes (IEEE style) so it never hides the waterfall tails
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=False,
              handlelength=2.0, columnspacing=0.9, borderaxespad=0.1)
    fig.tight_layout(pad=0.2)

    out_dir = os.path.join(ctm.ANALYSIS, args.tag, "paper")
    os.makedirs(ctm.LONG(out_dir), exist_ok=True)
    snr_lo, snr_hi = min(s for (_, _, s) in rows), max(s for (_, _, s) in rows)
    base = os.path.join(out_dir, f"bler_{args.tag}_s={'-'.join(map(str, args.seeds))}_snr{snr_lo}-{snr_hi}")
    # A PDF open in a viewer is locked on Windows - save under the next free _vN name instead.
    for n in range(1, 100):
        cand = base if n == 1 else f"{base}_v{n}"
        try:
            fig.savefig(ctm.LONG(cand + ".pdf"))
            fig.savefig(ctm.LONG(cand + ".png"), dpi=600)
            break
        except PermissionError:
            print(f"  [locked] {cand}.pdf/.png is open elsewhere - trying another name")
    base = cand
    print(f"data: {src}")
    print(f"config: {bases.pop()}")
    for m in modes:
        print(f"  {m:8s} <- {' + '.join(sorted(by_mode[m]))}")
    for n, miss in gaps:
        print(f"  [gap] {n}: no data at {miss} dB (line broken there)")
    if zeros:
        print("  BLER = 0 (line runs to the axis at the first zero of a run): " + ", ".join(f"{n}@{s}dB" for n, s in zeros))
    print("saved", base + ".pdf")
    print("saved", base + ".png")


if __name__ == "__main__":
    main()
