"""
build_analysis.py -- one-stop analysis page builder for DeepOFDM sweep results.

Usage:
    python build_analysis.py <tag>          e.g.  python build_analysis.py fasty
    python build_analysis.py                (no arg: lists available tags)

What it does, given experiment tag <tag>:
  1. Scans C:\\Projects\\Scratchpad for result CSVs of that tag and finds the
     unique configurations (all name fields except timestamp, seed and SNR).
  2. Creates C:\\Projects\\Scratchpad\\Analysis\\<tag>\\.
  3. Runs plot_multiple_csvs.plot_csvs() per configuration and moves each
     figure there as curve_<config>.png.
  4. Copies the per-SNR histogram/loss jpgs (lmmse/escnn/...) there under
     short names (long originals exceed Windows MAX_PATH at this depth).
  5. Writes <tag>.html: Part 1 = curves, Part 2 = histogram tables
     (rows = SNR, columns = detectors), each plot preceded by its parameters.
Safe to rerun: skips jpg copies that already exist, overwrites html/curves.
"""
import glob, os, re, shutil, sys

SCRATCH   = r"C:\Projects\Scratchpad"
UTILS     = r"C:\Projects\DeepOFDM\python_code\utils"
ANALYSIS  = os.path.join(SCRATCH, "Analysis")

LONG = lambda p: "\\\\?\\" + os.path.abspath(p)   # MAX_PATH-safe prefix

# ---------------------------------------------------------------- discovery
def base_csvs(tag):
    """BER csvs only (skip _bler/_mi companions) for the given tag."""
    out = []
    for p in glob.glob(os.path.join(SCRATCH, f"*_{tag}_s=*_SNR=*.csv")):
        n = os.path.basename(p)
        if n.endswith("_bler.csv") or n.endswith("_mi.csv"):
            continue
        out.append(n)
    return out

NAME_RE = re.compile(r"^(\d{8}_\d{4})_(.*)_s=(\d+)_SNR=(-?\d+)\.csv$")

def parse(n):
    m = NAME_RE.match(n)
    if not m:
        return None
    ts, key, seed, snr = m.groups()
    return key, int(snr)

def unique_configs(tag):
    """-> {config_key: sorted list of SNRs}"""
    cfgs = {}
    for n in base_csvs(tag):
        p = parse(n)
        if p:
            cfgs.setdefault(p[0], set()).add(p[1])
    return {k: sorted(v) for k, v in cfgs.items()}

def differing_tokens(keys):
    """Token positions whose value differs across configs (split on '_')."""
    tok_lists = [k.split("_") for k in keys]
    if len({len(t) for t in tok_lists}) != 1:
        # token counts differ; fall back to whole keys
        return {k: [k] for k in keys}
    n = len(tok_lists[0])
    diff_pos = [i for i in range(n) if len({t[i] for t in tok_lists}) > 1]
    if not diff_pos and len(keys) == 1:
        return {keys[0]: ["config"]}
    return {k: [k.split("_")[i] for i in diff_pos] for k in keys}

SAFE = lambda s: re.sub(r"[^A-Za-z0-9=.,\-]", "", s)

def sort_key(diffs):
    """Natural ordering, except tw follows presentation order 0.0, 1.0, 0.5."""
    TW_ORDER = {"0.0": 0, "1.0": 1, "0.5": 2}
    out = []
    for t in diffs:
        m = re.match(r"tw=(.+)$", t)
        if m:
            out.append(("tw", TW_ORDER.get(m.group(1), 9), m.group(1)))
        else:
            out.append((t,))
    return out

def _tw_has_no_effect(k):
    """True when tw can't have influenced this run: frz=a (fully frozen/loaded
    weights -- never trained) or tl=bce (bce loss doesn't read tw at all,
    only tsyn does -- see config.yaml's 'tw: ... # tsyn only')."""
    return bool(re.search(r"(^|_)frz=a(_|$)", k) or re.search(r"(^|_)tl=bce(_|$)", k))

def display_tokens(k, toks):
    """Tokens list for the visible h3 label. Drops tw=... when tw has no
    effect on the run (see _tw_has_no_effect)."""
    if _tw_has_no_effect(k):
        return [t for t in toks if not t.startswith("tw=")]
    return toks

def tw_note(k):
    """Explanatory note for the tw value found anywhere in the config key `k`.
    Suppressed when tw has no effect on the run (see _tw_has_no_effect)."""
    if _tw_has_no_effect(k):
        return ""
    m = re.search(r"(^|_)tw=([^_]+)(_|$)", k)
    if m:
        v = m.group(2)
        if v in ("0.0", "0"):
            return " &mdash; syndrome"
        if v in ("1.0", "1"):
            return " &mdash; tent"
        return f" &mdash; combined: {v}&middot;tent + {round(1-float(v), 2)}&middot;syndrome"
    return ""

def purge_redundant_tw_variants(tag):
    """Delete tw=1.0 result files where tw has no effect on training, so
    they're redundant duplicates of the tw=0.0 run: frz=a (frozen/loaded
    weights) and tl=bce (bce loss never reads tw). Scoped to this tag's
    files only."""
    patterns = [f"*frz=a*tw=1.0*_{tag}_s=*", f"*tl=bce*tw=1.0*_{tag}_s=*"]
    hits = set()
    for pat in patterns:
        hits.update(glob.glob(os.path.join(SCRATCH, pat)))
    for p in hits:
        os.remove(LONG(p))
    if hits:
        print(f"Purged {len(hits)} redundant tw=1.0 file(s) for tag '{tag}' (frz=a / tl=bce)")

# ---------------------------------------------------------------- building
def build(tag):
    purge_redundant_tw_variants(tag)
    cfgs = unique_configs(tag)
    if not cfgs:
        print(f"No CSVs found for tag '{tag}'"); return 1
    keys = sorted(cfgs)
    diffs = differing_tokens(keys)
    keys.sort(key=lambda k: sort_key(diffs[k]))

    out_dir = os.path.join(ANALYSIS, tag)
    os.makedirs(out_dir, exist_ok=True)

    # third plot panel is GFMI when nll=gf, plain BER otherwise
    third_panel = "GFMI" if all("nll=gf" in k for k in keys) else "BER"

    # detectors present among this tag's jpgs (token before first '_');
    # fixed presentation order: lmmse (reference) left, escnn right, others after
    found_dets = {os.path.basename(p).split("_", 1)[0]
                  for p in glob.glob(os.path.join(SCRATCH, f"*_{tag}_s=*_SNR=*.jpg"))}
    if not found_dets:  # source jpgs purged; fall back to already-copied ones
        found_dets = {os.path.basename(p).split("_", 1)[0]
                      for p in glob.glob(os.path.join(out_dir, "*_SNR*.jpg"))}
    DET_ORDER = {"lmmse": 0, "escnn": 1}
    dets = sorted(found_dets, key=lambda d: (DET_ORDER.get(d, 9), d))

    sys.path.insert(0, UTILS)
    import plot_multiple_csvs as pmc
    import matplotlib.pyplot as plt
    plt.show = lambda *a, **k: None

    html = ["""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>%s</title>
<style>
  body { font-family: Segoe UI, Arial, sans-serif; margin: 20px; max-width: 1500px; }
  h1 { font-size: 20px; } h3 { font-size: 15px; margin-bottom: 2px; }
  p.params { font-family: Consolas, monospace; font-size: 11px; color: #333; background: #f4f4f4; padding: 4px 8px; margin-top: 2px; word-break: break-all; }
  img.curve { width: 100%%; max-width: 1400px; border: 1px solid #ccc; margin-bottom: 10px; }
  table { border-collapse: collapse; width: 100%%; margin-bottom: 14px; }
  th, td { border: 1px solid #999; padding: 3px; text-align: center; vertical-align: top; }
  th { background: #eee; font-size: 13px; } td.snr { font-size: 14px; font-weight: bold; width: 3em; background: #f7f7f7; }
  td img { width: 100%%; max-width: 560px; }
</style></head><body>
<h1>%s &mdash; Part 1: BLER / MI / %s curves</h1>""" % (tag, tag, third_panel)]

    failed, copied = [], 0
    for k in keys:
        short = SAFE("_".join(diffs[k]))
        label = " | ".join(display_tokens(k, diffs[k]))
        pattern = f"*_{k}_s=*_SNR=*.csv"
        note = tw_note(k)
        print("=" * 70); print(f"[{label}]  {pattern}")
        has_ue_curve = False
        try:
            pmc.plot_csvs(pattern)
            src = os.path.join(SCRATCH, "plot_output.png")
            if os.path.exists(src):
                shutil.move(src, os.path.join(out_dir, f"curve_{short}.png"))
                print("  [OK] curve")
            else:
                raise RuntimeError("plot_output.png not produced")
            # Second figure with per-UE lines only -- produced only when the
            # CSVs have per-user columns (see plot_multiple_csvs.py).
            src_ue = os.path.join(SCRATCH, "plot_output_peruser.png")
            if os.path.exists(src_ue):
                shutil.move(src_ue, os.path.join(out_dir, f"curve_{short}_peruser.png"))
                has_ue_curve = True
        except Exception as e:
            print(f"  [FAIL] {e}"); failed.append((label, str(e)))
        plt.close("all")
        html.append(f"<h3>{label}{note}</h3><p class='params'>{k}</p>"
                    f"<img class='curve' src='curve_{short}.png'>")
        if has_ue_curve:
            html.append(f"<img class='curve' src='curve_{short}_peruser.png'>")

    html.append(f"<h1>{tag} &mdash; Part 2: LLR histograms &amp; training loss "
                f"({' | '.join(d.upper() for d in dets)})</h1>")

    def jpg_snrs(k):
        """SNRs for which histogram jpgs exist for this config (any detector)."""
        toks = diffs[k]
        snrs = set()
        for det in dets:
            pat = os.path.join(SCRATCH, f"{det}_2*_{k}_s=*_SNR=*.jpg")
            for p in glob.glob(pat):
                m = re.search(r"_SNR=(-?\d+)\.jpg$", os.path.basename(p))
                if m:
                    snrs.add(int(m.group(1)))
            # already-copied short-named jpgs also count (source may be purged)
            for p in glob.glob(os.path.join(ANALYSIS, tag, f"{det}_{SAFE('_'.join(toks))}_SNR*.jpg")):
                m = re.search(r"_SNR(-?\d+)\.jpg$", os.path.basename(p))
                if m:
                    snrs.add(int(m.group(1)))
        return sorted(snrs)

    missing = []
    for k in keys:
        short = SAFE("_".join(diffs[k]))
        label = " | ".join(display_tokens(k, diffs[k]))
        html.append(f"<h3>{label}{tw_note(k)}</h3><p class='params'>{k}</p>")
        snrs_avail = jpg_snrs(k)
        if not snrs_avail:
            html.append("<p class='params'>no histogram jpgs found for this configuration</p>")
            continue
        html.append("<table><tr><th>SNR (dB)</th>" +
                    "".join(f"<th>{d.upper()}</th>" for d in dets) + "</tr>")
        for snr in snrs_avail:
            row = [f"<td class='snr'>{snr}</td>"]
            for det in dets:
                sname = f"{det}_{short}_SNR{snr}.jpg"
                dst = os.path.join(out_dir, sname)
                if not os.path.exists(dst):
                    pat = os.path.join(SCRATCH, f"{det}_2*_{k}_s=*_SNR={snr}.jpg")
                    hits = glob.glob(pat)
                    if hits:
                        shutil.copy2(LONG(hits[0]), dst); copied += 1
                if os.path.exists(dst):
                    row.append(f"<td><img src='{sname}'></td>")
                else:
                    row.append("<td>missing</td>"); missing.append((det, label, snr))
            html.append("<tr>" + "".join(row) + "</tr>")
        html.append("</table>")

    html.append("</body></html>")
    out_html = os.path.join(out_dir, f"{tag}.html")
    with open(out_html, "w", encoding="utf-8") as f:
        f.write("\n".join(html))

    print("=" * 70)
    print(f"Configs: {len(keys)} | curves failed: {len(failed)} | jpgs copied: {copied} | cells missing: {len(missing)}")
    for x in failed:  print("  CURVE FAIL:", x)
    for x in missing: print("  MISSING:", x)
    print(f"Open: {out_html}")
    return 0

# ---------------------------------------------------------------- main
def detect_tags():
    return sorted({m.group(1) for n in glob.glob(os.path.join(SCRATCH, "*_s=*_SNR=*.csv"))
                   for m in [re.search(r"_([A-Za-z][A-Za-z0-9]*)_s=\d+_SNR=", os.path.basename(n))] if m})

if __name__ == "__main__":
    tags = sys.argv[1:] if len(sys.argv) > 1 else detect_tags()
    if not tags:
        print("No result CSVs found in", SCRATCH); sys.exit(1)
    print("Building tags:", ", ".join(tags))
    rc = 0
    for t in tags:
        rc |= build(t)
    sys.exit(rc)
