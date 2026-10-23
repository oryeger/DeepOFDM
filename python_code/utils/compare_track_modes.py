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
per-SNR table is also written as CSV next to the figure (it is not printed). The figure is also opened in
Paint (plot_drift_log's helper) unless --no-open is given.
"""
import argparse, collections, glob, io, os, re, sys, tarfile

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:   # run as a script from DeepOFDM root -> sys.path[0] is this utils dir
    from plot_drift_log import _open_in_viewer
except ImportError:   # run as python -m python_code.utils.compare_track_modes
    from python_code.utils.plot_drift_log import _open_in_viewer

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
# Per seed, fill SNRs that other seeds of the same curve have but this seed is missing - interior
# gaps only (between this seed's own first and last SNR; never extrapolated), before averaging, so
# neighbouring points don't average different seed subsets. log10 interpolation for BLER/BER (linear
# when a bracketing value is 0), linear for MI. Same idea as plot_multiple_csvs'
# INTERPOLATE_MISSING_PER_SEED. Only gaps of at most MAX_INTERP_GAP consecutive missing SNRs are
# filled, from that seed's own neighbours (1: a missing X is filled only when X-1 and X+1 exist); a
# longer gap stays open, so the seed counts as incomplete (see DROP_INCOMPLETE_SEEDS).
INTERPOLATE_MISSING_PER_SEED = True
MAX_INTERP_GAP = None   # None: no cap - fill any interior gap (as before); an int N: only gaps of <= N missing SNRs
# After that fill: the SNR grid is the SNRs that more than half of the figure's seeds have in ANY mode
# (with several seeds, a mode is not required at an SNR no seed ran it at).
# A seed still missing any grid SNR of ANY mode (gaps at its ends, which are never extrapolated, or
# no runs of that mode at all) is left out of EVERY mode's average, so all curves average the same
# seed set and stay comparable; SNRs outside a mode's grid are not drawn. Decided on the BLER CSVs,
# applied to every panel. The kept/dropped seeds go into the figure title. If no seed is complete
# (e.g. a single seed missing points of a listed mode), nothing is drawn for that table.
DROP_INCOMPLETE_SEEDS = True


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


def _interp_interior(have, want, log):
    """have: {snr: value} of one seed. -> {snr: value} for the SNRs in `want` strictly inside
    (min(have), max(have)) that `have` lacks, interpolated from the nearest neighbours."""
    xs = sorted(have)
    out = {}
    if len(xs) < 2:
        return out
    for s in want:
        if s in have or not xs[0] < s < xs[-1]:
            continue
        i = next(j for j, x in enumerate(xs) if x > s)
        x0, x1, y0, y1 = xs[i - 1], xs[i], have[xs[i - 1]], have[xs[i]]
        if MAX_INTERP_GAP is not None and sum(1 for w in want if x0 < w < x1) > MAX_INTERP_GAP:   # gap too long - leave it open
            continue
        a = (s - x0) / (x1 - x0)
        if log and y0 > 0 and y1 > 0:
            out[s] = 10 ** ((1 - a) * np.log10(y0) + a * np.log10(y1))
        else:
            out[s] = (1 - a) * y0 + a * y1
    return out


def _why_txt(sd, why, have_n):
    """'ekfbp 15/31 SNRs ekfht 29/31 SNRs': per incomplete mode of seed sd, how many of the grid's
    SNRs it has (an interpolated point counts as had)."""
    return " ".join(f"{m} {have_n[(sd, m)][0]}/{have_n[(sd, m)][1]} SNRs" if (sd, m) in have_n else m
                    for m in why[sd])


def aggregate(store, chosen, modes, title="", seeds=None, snr_range=None, incomplete="seeds"):
    """The seed averaging behind make_figure (and write_table): per mode/metric/SNR the mean over the
    CSV rows, then over the kept seeds. `modes`: legend labels, in order; chosen[label] is the
    config key each one plots, from a load() store. Any number of modes (no palette limit here).
    -> dict(data, modes, all_modes, title, ref_name, rate, n_interp); data[mode][metric][snr] =
    (escnn, ref, n_seeds). snr_range=(lo, hi) keeps only SNRs in [lo, hi] (inclusive).
    chosen[label] may also be a tuple of config keys, pooled together (e.g. one per seed when each
    seed's run carries its own w=/r= hash).
    incomplete (DROP_INCOMPLETE_SEEDS): what to leave out when a seed doesn't cover
    a mode's SNR grid - "seeds": that seed, from every mode (all modes shown, fewer seeds); "modes":
    that mode (every seed kept, only fully covered modes shown)."""
    all_modes = list(modes)   # color slots stay fixed even when incomplete="modes" hides some
    cfgs = lambda v: (v,) if isinstance(v, str) else tuple(v)

    # aggregate: data[mode][metric][snr] -> (escnn, ref, n_seeds); ref = primary receiver (see AUG_REF)
    data = {m: {met: {} for met, *_ in METRICS} for m in modes}
    # Pass 1: per (mode, metric), every seed's curve, interior gaps filled (INTERPOLATE_MISSING_PER_SEED).
    curves = {m: {} for m in modes}    # mode -> met -> {seed: {snr: (escnn, ref)}}
    filled = {m: {} for m in modes}    # mode -> {seed: SNRs of its interpolated BLER points}
    # SNRs to fill toward: what ANY mode of the figure has (any seed), so a lone seed's holes are filled too
    all_cfgs = {c for v in chosen.values() for c in cfgs(v)}
    want_all = {kind: sorted({snr for (c, k, seed, snr) in store
                              if c in all_cfgs and k == kind and (seeds is None or seed in seeds)
                              and (snr_range is None or snr_range[0] <= snr <= snr_range[1])})
                for _, kind, _, _ in METRICS}
    for mode, config in chosen.items():
        for met, kind, scale, agg in METRICS:
            per_seed = collections.defaultdict(dict)
            for (c, k, seed, snr), (ts, e, l, _) in store.items():
                if c in cfgs(config) and k == kind and (seeds is None or seed in seeds)                         and (snr_range is None or snr_range[0] <= snr <= snr_range[1]):
                    per_seed[seed][snr] = (e.mean(), l.mean())
            if INTERPOLATE_MISSING_PER_SEED:
                want = want_all[kind]
                for seed, d in per_seed.items():
                    fe = _interp_interior({s_: v[0] for s_, v in d.items()}, want, scale == "log")
                    fl = _interp_interior({s_: v[1] for s_, v in d.items()}, want, scale == "log")
                    for s_ in fe:
                        d[s_] = (fe[s_], fl[s_])
                    if met == "BLER":
                        filled[mode][seed] = set(fe)
            curves[mode][met] = dict(per_seed)

    # Pass 2 (DROP_INCOMPLETE_SEEDS): one seed set for the whole figure, so the modes stay comparable.
    # The grid is figure-wide: the BLER SNRs that more than half of the seeds have in ANY mode. Judging a
    # mode only against its own seeds let a gap shared by half the seeds (e.g. 1 of 2) shrink that mode's
    # grid instead of flagging the seeds. With several seeds a mode is not required at SNRs no seed ran
    # it at; with one seed every mode must cover the figure-wide grid. A seed missing any grid SNR of ANY
    # mode (or that mode altogether) is dropped from every mode; SNRs off a mode's grid are not drawn.
    # If that leaves no seed (e.g. the only seed is incomplete), nothing is drawn for the table.
    all_seeds = sorted({sd for m in modes for d in curves[m].values() for sd in d})
    keep, grids, why = set(all_seeds), {}, collections.defaultdict(list)
    have_n = {}   # (seed, mode) -> (grid SNRs the seed has, grid size), for incomplete pairs
    if DROP_INCOMPLETE_SEEDS and all_seeds:
        fig_cover = collections.Counter(
            snr for sd in all_seeds
            for snr in set().union(*(curves[m]["BLER"].get(sd, {}).keys() for m in modes)))
        fig_grid = {snr for snr, n in fig_cover.items() if n > len(all_seeds) / 2}
        for mode in modes:
            bl = curves[mode]["BLER"]
            ran = {snr for d in bl.values() for snr in d}
            grids[mode] = fig_grid if len(all_seeds) == 1 else fig_grid & ran
            for sd in all_seeds:
                if not grids[mode] <= set(bl.get(sd, {})):
                    keep.discard(sd)
                    why[sd].append(mode)
                    have_n[(sd, mode)] = (len(grids[mode] & set(bl.get(sd, {}))), len(grids[mode]))
        if incomplete == "modes":
            bad = {m for ms in why.values() for m in ms}
            if bad == set(modes):
                print("  [WARN] no mode is covered by every seed - showing every mode, averaging every available seed per point")
                grids = {}
            else:
                print(f"  every seed kept; left out mode(s) not covered by every seed: "
                      + ", ".join(f"{m} (s={','.join(str(sd) for sd in all_seeds if m in why[sd])})"
                                  for m in modes if m in bad))
                title = f"{title}  [all {len(all_seeds)} seeds; modes left out: {', '.join(m for m in modes if m in bad)}]"
                modes = [m for m in modes if m not in bad]
            keep, why = set(all_seeds), collections.defaultdict(list)
        elif not keep:   # no seed covers every mode's grid (e.g. the only seed is incomplete) - draw nothing
            print("  [WARN] no seed covers every mode's SNR grid - nothing is drawn for this table")
    dropped = sorted(set(all_seeds) - keep)
    absent = sorted(set(seeds) - set(all_seeds)) if seeds else []   # requested, but no runs in the data at all
    drop_txt = {sd: "incomplete: " + _why_txt(sd, why, have_n) for sd in dropped}
    drop_txt.update({sd: "no runs found" for sd in absent})
    dropped = sorted(dropped + absent)
    # only interpolated points that are actually drawn (kept seed, on the mode's grid)
    n_interp = {m: sum(len(s if m not in grids else s & grids[m]) for sd, s in filled[m].items() if sd in keep)
                for m in modes}
    interp_pts = {m: {sd: sorted(s if m not in grids else s & grids[m]) for sd, s in filled[m].items()
                      if sd in keep and (s if m not in grids else s & grids[m])} for m in modes}

    # aggregate: data[mode][metric][snr] -> (escnn, ref, n_seeds); ref = primary receiver (see AUG_REF)
    data = {m: {met: {} for met, *_ in METRICS} for m in modes}
    for mode in modes:
        for met, *_ in METRICS:
            per_snr = collections.defaultdict(list)
            for sd, d in curves[mode][met].items():
                if sd not in keep:
                    continue
                for snr, v in d.items():
                    if mode not in grids or snr in grids[mode]:
                        per_snr[snr].append(v)
            for snr, v in per_snr.items():
                data[mode][met][snr] = (np.mean([x[0] for x in v]), np.mean([x[1] for x in v]), len(v))
    if dropped:
        print(f"  averaged {len(keep)}/{len(all_seeds) + len(absent)} seeds in every mode; dropped "
              + ", ".join(f"s={sd} ({drop_txt[sd]})" for sd in dropped))
        title = f"{title}  [seeds {','.join(map(str, sorted(keep)))}; dropped {','.join(map(str, dropped))}]"
    # code rate from the R=<r> token (same for all modes after --filter, else taken from the first)
    rate = next((float(t[2:]) for t in cfgs(chosen[modes[0]])[0].split("_") if t.startswith("R=")), None)

    # reference name from the first mode's CSVs (falls back to LMMSE per file - see _record)
    ref_names = {v[3] for (c, *_), v in store.items() if c in cfgs(chosen[modes[0]])}
    ref_name = ref_names.pop() if len(ref_names) == 1 else "LMMSE"
    return dict(data=data, modes=modes, all_modes=all_modes, title=title, ref_name=ref_name,
                rate=rate, n_interp=n_interp, kept_seeds=sorted(keep), dropped_seeds=dropped,
                dropped_why={sd: why[sd] for sd in dropped}, interp_pts=interp_pts,
                dropped_txt=drop_txt,
                snr_grid=sorted(fig_grid) if DROP_INCOMPLETE_SEEDS and all_seeds else [])


def write_seeds_note(agg, csv_path, config=None):
    """<csv>_seeds.txt next to a table: which seeds its averages use (read by
    escnn_matlab\\master_plot_trk_bler_paper.m, which prints it). config (optional): the
    configuration key shared by the table's modes - MATLAB passes its tokens back as --filter to
    re-average just the modes/seeds it draws."""
    path = os.path.splitext(csv_path)[0] + "_seeds.txt"
    with open(LONG(path), "w") as f:
        f.write("averaged seeds: " + (" ".join(map(str, agg["kept_seeds"])) or "- (nothing drawn)") + "\n")
        f.write("dropped seeds: " + (", ".join(f"{sd} ({agg.get('dropped_txt', {}).get(sd) or 'incomplete: ' + ' '.join(agg['dropped_why'][sd])})"
                                               for sd in agg["dropped_seeds"]) or "-") + "\n")
        left = [m for m in agg.get("all_modes", []) if m not in agg["modes"]]
        if left:
            f.write("left out modes (incomplete): " + " ".join(left) + "\n")
        pts = [f"s={sd} {m}@" + ",".join(f"{x:g}" for x in snrs)
               for m, d in agg.get("interp_pts", {}).items() for sd, snrs in sorted(d.items())]
        f.write("interpolated: " + ("; ".join(pts) or "-") + "\n")
        if config:
            f.write("config: " + config + "\n")


def table_rows(agg):
    """Per-SNR rows of an aggregate() result: SNR, <ref>_BLER, then <mode>_BLER/_MI/_BER per mode
    (NaN where a mode has no point) - the CSV escnn_matlab\\master_plot_trk_bler_paper.m reads."""
    data, modes, ref_name = agg["data"], agg["modes"], agg["ref_name"]
    # no kept seed -> all-NaN rows on the figure's SNR grid, so the table (and its subplot) is empty
    grid = sorted({s for m in data.values() for d in m.values() for s in d}) or agg.get("snr_grid", [])
    rows = []
    for s in grid:
        row = {"SNR": s, f"{ref_name}_BLER": data[modes[0]]["BLER"].get(s, (np.nan,) * 3)[1]}
        for mode in modes:
            for met, *_ in METRICS:
                row[f"{mode}_{met}"] = data[mode][met].get(s, (np.nan,) * 3)[0]
        rows.append(row)
    return rows


def write_table(store, chosen, modes, path, config=None, **kw):
    """aggregate() + table_rows() -> CSV at `path`, without drawing (so any number of modes).
    config: written into the seeds note (see write_seeds_note). kw: aggregate's
    title/seeds/snr_range/incomplete. Returns the aggregate() dict."""
    agg = aggregate(store, chosen, modes, **kw)
    if not agg["modes"]:   # incomplete="modes" and no mode is covered by every seed
        print("  no table: no mode is complete for every seed -", path)
        return agg
    os.makedirs(LONG(os.path.dirname(path)), exist_ok=True)
    pd.DataFrame(table_rows(agg)).to_csv(LONG(path), index=False)
    write_seeds_note(agg, path, config)
    print("table", path)
    return agg


def make_figure(store, chosen, modes, title, out, seeds=None, table=False, snr_range=None, incomplete="seeds"):
    """Draw the BLER/MI/BER comparison of `modes` (legend labels, in order; chosen[label] is the
    config key each one plots) from a load() store, save it to `out` and return the figure.
    Averaging: see aggregate(). Labels found in MODE_SLOT keep their fixed color; at most
    len(COLORS) modes. With table=True the per-SNR table is also written as CSV next to the figure."""
    a = aggregate(store, chosen, modes, title, seeds, snr_range, incomplete)
    data, modes, all_modes, title = a["data"], a["modes"], a["all_modes"], a["title"]
    ref_name, rate, n_interp = a["ref_name"], a["rate"], a["n_interp"]
    cfgs = lambda v: (v,) if isinstance(v, str) else tuple(v)

    grid = sorted({s for m in data.values() for d in m.values() for s in d})
    step = min(np.diff(grid)) if len(grid) > 1 else 0
    slot = mode_slots(all_modes)
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
            if n_interp[mode]:
                lab += f"  [{n_interp[mode]} interp]"
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

    rows = table_rows(a)
    if table:
        tpath = os.path.splitext(out)[0] + ".csv"
        pd.DataFrame(rows).to_csv(LONG(tpath), index=False)
        write_seeds_note(a, tpath)
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
                    help="do not open the figure in Paint")
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
        # Configs that differ only in the per-seed weight hash (r=/w=, one per seed's checkpoint) are the
        # same curve: pool them (make_figure accepts a tuple of config keys).
        if len(c) > 1 and len({re.sub(r"_(r|w)=[^_]+", "", x) for x in c}) == 1:
            chosen[mode] = tuple(c)
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
            _open_in_viewer(fig)
        except Exception as e:
            print(f"could not open image viewer: {e}")


if __name__ == "__main__":
    main()
