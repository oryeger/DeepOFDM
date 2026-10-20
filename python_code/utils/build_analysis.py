"""
build_analysis.py -- one-stop analysis page builder for DeepOFDM sweep results.

Usage:
    python build_analysis.py <tag>          e.g.  python build_analysis.py fasty
    python build_analysis.py                (no arg: lists available tags)

What it does, given experiment tag <tag>:
  1. Scans C:\\Projects\\Scratchpad for result CSVs of that tag and finds the
     unique configurations (all name fields except timestamp and SNR - each seed is its own
     configuration, keyed '<config>_s=<seed>'; w=/r= hashes are left out of the labels when
     they only shadow the seed).
  2. Creates C:\\Projects\\Scratchpad\\Analysis\\<tag>\\.
  3. Runs plot_multiple_csvs.plot_csvs() per configuration and moves each
     figure there as curve_<config>.png.
  4. Copies the per-SNR histogram/loss jpgs (lmmse/escnn/...) there under
     short names (long originals exceed Windows MAX_PATH at this depth).
  5. Runs compare_track_modes.make_figure() once per unique configuration that was run with
     several tracking modes (t=<mode>; configs are grouped by every token except the
     tracking-mode family CMP_FAMILY), overlaying all of its modes on one BLER/MI/BER figure,
     and writes the per-SNR table behind each such section (all its modes) as compare_*.csv -
     the input of escnn_matlab\\master_plot_trk_bler_paper.m (seed-averaged: compare_avg_<i>.csv).
  6. Writes <tag>.html: Part 0 = tracking-mode comparisons, Part 1 = curves, Part 2 = histogram
     tables (rows = SNR, columns = detectors), each plot preceded by its parameters.
A tag mixing several code rates / primary receivers gets one build per combination, as subdirs
of a parent Analysis\\<tag>\\ (e.g. icpilot\\icpilot_R=0.59\\), with an index.html linking them.
Rerunning never overwrites a previous build: the newest build always lands at
C:\\Projects\\Scratchpad\\Analysis\\<tag>\\ (no suffix). If that already exists, it - and any
older <tag>_N builds - are shifted up by one index first (<tag> -> <tag>_2, <tag>_2 -> <tag>_3,
etc., highest index first so renames never collide), so old builds stick around (just
renumbered) until you delete them yourself.
"""
import collections, glob, os, re, shutil, sys

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

def parse_seeded(n):
    """-> (config key with its seed, '<config>_s=<seed>', snr), or None."""
    m = NAME_RE.match(n)
    if not m:
        return None
    ts, key, seed, snr = m.groups()
    return f"{key}_s={seed}", int(snr)

def _split_seed(k):
    """'<config>_s=<seed>' -> (config, seed)."""
    config, seed = k.rsplit("_s=", 1)
    return config, int(seed)

def unique_configs(tag):
    """-> {'<config>_s=<seed>': sorted list of SNRs} - seeds are kept apart, not averaged."""
    cfgs = {}
    for n in base_csvs(tag):
        p = parse_seeded(n)
        if p:
            cfgs.setdefault(p[0], set()).add(p[1])
    return {k: sorted(v) for k, v in cfgs.items()}

def _split_key_tokens(k):
    """Split a config key into (name, token) pairs, in order. 'name=value' tokens use the
    part before '=' as name; bare tokens (no '=', e.g. the leading channel/mod text or the
    trailing cur_str) get a positional name "bare_N" instead - stable across keys even when
    the number of optional name=value tokens between them differs (see differing_tokens)."""
    pairs = []
    bare_idx = 0
    for tok in k.split("_"):
        if "=" in tok:
            name = tok.split("=", 1)[0]
        else:
            name, bare_idx = f"bare_{bare_idx}", bare_idx + 1
        pairs.append((name, tok))
    return pairs

def differing_tokens(keys):
    """Tokens whose value differs across configs, matched by token *name* (see
    _split_key_tokens) rather than raw '_'-split position - so a token only some configs
    have (e.g. an older CSV predating a newly added filename tag) still shows up as a real
    difference instead of falling back to the whole key for every config."""
    parsed = {k: _split_key_tokens(k) for k in keys}
    names_in_order = []
    seen = set()
    for k in keys:
        for name, _ in parsed[k]:
            if name not in seen:
                seen.add(name)
                names_in_order.append(name)
    value_by_name = {name: {k: dict(parsed[k]).get(name) for k in keys} for name in names_in_order}
    diff_names = {name for name in names_in_order if len(set(value_by_name[name].values())) > 1}
    if not diff_names and len(keys) == 1:
        return {keys[0]: ["config"]}
    return {k: [tok for name, tok in parsed[k] if name in diff_names] for k in keys}

# Per-run hash tokens (w= saved-weights tag, r= loaded-weights tag): random, so they usually just
# shadow the seed. Hidden from labels whenever they don't distinguish anything else.
HASH_TOKENS = ("w", "r")

def drop_redundant_hashes(diffs):
    """Remove HASH_TOKENS from each config's differing-token list when the remaining tokens
    still tell every config apart (e.g. w= differs only because each seed saved its own weights)."""
    for hname in HASH_TOKENS:
        trimmed = {k: [t for t in toks if not t.startswith(hname + "=")] for k, toks in diffs.items()}
        if trimmed != diffs and len({tuple(t) for t in trimmed.values()}) == len(diffs)                 and all(trimmed.values()):
            diffs = trimmed
    return diffs

def _hash_tag(k):
    """The weights hashes (r= loaded / w= saved, 6-char SHA1 prefixes) of a config key, e.g. 'r=fdc7ba'."""
    return " ".join(tok for name, tok in _split_key_tokens(k) if name in HASH_TOKENS)

def _h3(label, k):
    """Heading for one config: its differing tokens, plus its weights hash when the label lacks it."""
    h = _hash_tag(k)
    extra = f" <span class='wts'>[weights {h}]</span>" if h and h not in label else ""
    return f"<h3>{label}{extra}</h3>"

SAFE = lambda s: re.sub(r"[^A-Za-z0-9=_.,\-]", "", s)

# Token names that are intrinsic to the tracking mode itself (ekf.py only
# emits csg/ep/bs/loss/tw/po for specific track_modes; lr is always emitted right
# after trk) rather than independent sweep axes -- see ekf.py's
# get_escnn_title_string, ~line 272-282.
# "t" is the tracking-mode token since 2026-10-04 (older names use "trk"); both are accepted.
TRK_FAMILY = {"trk", "t", "csg", "lr", "ep", "bs", "loss", "tw", "po"}
# Grouping for the Part 0 tracking-mode comparisons additionally ignores bpi (ekf.py emits it only
# for the BP modes ekfbp/ekfht/sgdsbp/...). Kept out of TRK_FAMILY so mat_target's buckets don't move.
CMP_FAMILY = TRK_FAMILY | {"bpi"}

def mat_target(tag, k, diffs_k, short, aug_type):
    """Where plot_csvs() should save this config's .mat file(s).

    Configs without a trk= token are left alone entirely (returns (None, None,
    None) so plot_csvs() falls through to its own default:
    CSV_DIR/mat_files/<aug_type>.mat) -- untouched, so anything depending on
    that path (e.g. master_plot_mi_bler.m's algorithm-comparison workflow)
    keeps working exactly as before, bugs and all.

    Configs with a trk= token are bucketed into a shared subfolder per
    *non-trk-family* config (so an ekf/sgdbce/sgdbcei/sgdsyn/notrack sweep of
    the same underlying config lands together even though e.g. sgdbcei alone
    carries an extra csg= token), with each trk mode saved as
    <aug_type>_<weights_track_mode>.mat (e.g. 'lmmse_sgdbcei.mat', not the
    full 'trk=sgdbceicsg=1lr=5e-3ep=1loss=bce.mat') -- ready for a
    trk-comparison plot to glob that one folder. This is a new axis (trk
    sweeps didn't have a working save path before -- every config would
    overwrite the same mat_files/lmmse.mat), so nothing existing depends on
    it yet.

    aug_type comes from plot_multiple_csvs.detect_aug_type(k) (the same
    which_augment detection plot_csvs() itself uses for the legacy path, kept
    as one source of truth) so trk sweeps across different augment types
    don't collide even when 'aug=' isn't itself a differing token in this
    build.

    NOTE: two configs in the same non-trk-family bucket that share both
    aug_type and trk mode but differ only in one of csg/lr/ep/loss/tw (e.g. a
    learning-rate sweep at fixed trk=sgdbce) WILL still collide and overwrite
    -- build() checks for and warns about this (see `seen_mat_targets`
    below); it does not happen for any config in the repo today.

    Also returns a clean trk_mode label ('ekf', 'sgdbcei', ...) for
    mat_extra, stored as S.trk_mode inside the .mat itself so a MATLAB reader
    always has it available even where it's not the filename.

    Returns: (mat_dir, mat_name, trk_mode) -- all None for a config with no
    trk= token.
    """
    pairs = _split_key_tokens(k)
    trk_toks = [tok for name, tok in pairs if name in TRK_FAMILY]
    if not trk_toks:
        return None, None, None
    rest_toks = [tok for name, tok in pairs if name not in TRK_FAMILY and tok in diffs_k]
    mode_toks = [tok for name, tok in pairs if name in ("trk", "t")]
    trk_mode = (mode_toks[0] if mode_toks else trk_toks[0]).split("=", 1)[1]
    mat_dir  = os.path.join(SCRATCH, "mat_files", tag, SAFE("_".join(rest_toks)) or "base")
    mat_name = SAFE(f"{aug_type.lower()}_{trk_mode}")
    return mat_dir, mat_name, trk_mode

def _trk_group(k):
    """Presentation rank of a config's tracking mode: notrack, ekf, ekfi (incl. the ekfi1..ekfi6
    debug ladder), sgdbcei, ekfbp (incl. ekfbps/ekfbpc/ekfcrc/ekfht), sgdsbp (incl. sgdht), then everything else
    (other modes, e.g. ekfibp, and configs with no t=/trk= token)."""
    mode = next((tok.split("=", 1)[1] for name, tok in _split_key_tokens(k) if name in ("trk", "t")), None)
    if mode is None:
        return 6
    if re.fullmatch(r"ekfi\d*", mode) or mode in ("ekfit", "ekfis"):
        return 2
    if mode in ("ekfbp", "ekfbps", "ekfbpc", "ekfcrc", "ekfht"):
        return 4
    return {"notrack": 0, "ekf": 1, "sgdbcei": 3, "sgdbceit": 3, "sgdsbp": 5, "sgdht": 5}.get(mode, 6)

def sort_key(diffs):
    """Natural ordering, except tw follows presentation order 0.0, 1.0, 0.5 and seeds (s=) sort
    numerically."""
    TW_ORDER = {"0.0": 0, "1.0": 1, "0.5": 2}
    out = []
    for t in diffs:
        m = re.match(r"tw=(.+)$", t)
        if m:
            out.append(("tw", TW_ORDER.get(m.group(1), 9), m.group(1)))
        elif re.fullmatch(r"s=\d+", t):
            out.append(("s=", int(t[2:])))
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

def _rotate_out_dir(tag):
    """The newest build always lands at ANALYSIS/<tag> (no suffix). If that's taken, it - and
    any older ANALYSIS/<tag>_N builds - are shifted up by one index first (<tag> -> <tag>_2,
    <tag>_2 -> <tag>_3, ...), highest index first so the renames never collide - so a previous
    build is never overwritten, just renumbered."""
    base = os.path.join(ANALYSIS, tag)
    if not os.path.isdir(base):
        return base
    idx = 2
    while os.path.isdir(os.path.join(ANALYSIS, f"{tag}_{idx}")):
        idx += 1
    for n in range(idx - 1, 1, -1):
        os.rename(os.path.join(ANALYSIS, f"{tag}_{n}"), os.path.join(ANALYSIS, f"{tag}_{n + 1}"))
    os.rename(base, os.path.join(ANALYSIS, f"{tag}_2"))
    print(f"Analysis dir for tag '{tag}' already existed; rotated it (and any older builds) up "
          f"by one index, so this build gets the unsuffixed '{tag}' dir")
    return base

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
def _code_rate(k):
    """Value of the R=<code rate> token (e.g. '0.48'), or None if the key has none."""
    return next((tok.split("=", 1)[1] for name, tok in _split_key_tokens(k) if name == "R"), None)

def _primary(k):
    """Primary receiver (which_augment) of a config: 'LMMSE'/'SPHERE'/'DEEPSIC'/'DEEPRX', or
    'NOAUG' when the key carries no recognised aug token (e.g. NO_AUGMENT)."""
    sys.path.insert(0, UTILS)
    import plot_multiple_csvs as pmc
    return pmc.detect_aug_type(k) or "NOAUG"

def build(tag):
    """One build per (code rate, primary receiver): a batch mixing several R= values and/or
    several primary receivers gets a parent ANALYSIS/<tag>/ dir (index.html linking the builds)
    holding a <tag>_R=<rate>_<PRIMARY>/ subdir + html per combination (only the axes that actually
    vary go into the suffix); the parent is what gets rotated (<tag> -> <tag>_2, ...). A batch with
    a single rate and primary keeps the plain ANALYSIS/<tag>/<tag>.html layout."""
    purge_redundant_tw_variants(tag)
    cfgs = unique_configs(tag)
    if not cfgs:
        print(f"No CSVs found for tag '{tag}'"); return 1
    groups = {}
    for k in cfgs:
        groups.setdefault((_code_rate(k), _primary(k)), []).append(k)
    multi_rate = len({r for r, _ in groups}) > 1
    multi_prim = len({a for _, a in groups}) > 1
    if len(groups) == 1:
        return _build_one(tag, tag, list(cfgs))
    def suffix(rate, prim):
        return "".join(([f"_R={rate}"] if multi_rate else []) + ([f"_{prim}"] if multi_prim else []))
    order = sorted(groups, key=lambda g: (str(g[0]), g[1]))
    print(f"Tag '{tag}' mixes {len(groups)} code-rate/primary combinations -> one build each: "
          + ", ".join(suffix(*g).lstrip("_") for g in order))
    parent = _rotate_out_dir(tag)
    os.makedirs(parent, exist_ok=True)
    names = [f"{tag}{suffix(*g)}" for g in order]
    rc = 0
    for g, name in zip(order, names):
        rc |= _build_one(tag, name, groups[g], out_dir=os.path.join(parent, name))
    with open(os.path.join(parent, "index.html"), "w", encoding="utf-8") as f:
        f.write(f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>{tag}</title></head>"
                f"<body style='font-family:sans-serif'><h2>{tag}</h2><ul>"
                + "".join(f"<li><a href='{n}/{n}.html'>{n}</a></li>" for n in names)
                + "</ul></body></html>")
    print(f"Index of the {len(names)} builds: {os.path.join(parent, 'index.html')}")
    return rc

def _mode(k):
    """Tracking mode (value of the t=/trk= token) of a config, or None."""
    return next((tok.split("=", 1)[1] for name, tok in _split_key_tokens(k) if name in ("trk", "t")), None)

def _cmp_entries(keys, drop):
    """Group `keys` for the Part 0 comparisons: group key = every token except the CMP_FAMILY ones
    and the token names in `drop`. Keeps groups run with >= 2 tracking modes. -> {gkey: {legend
    label: [keys]}}; label = the mode, plus its other trk-family tokens (lr=, csg=, ...) when the
    same mode appears with several of them in a group."""
    groups = {}
    for k in keys:
        if _mode(k) is None:
            continue
        gkey = "_".join(tok for name, tok in _split_key_tokens(k) if name not in CMP_FAMILY | drop)
        groups.setdefault(gkey, []).append(k)
    out = {}
    for gkey, ks in groups.items():
        variants = collections.defaultdict(set)   # mode -> its distinct trk-family token tuples
        extra = {}
        for k in ks:
            extra[k] = tuple(tok for name, tok in _split_key_tokens(k) if name in CMP_FAMILY - {"t", "trk"})
            variants[_mode(k)].add(extra[k])
        labeled = collections.defaultdict(list)
        for k in ks:
            lab = _mode(k)
            if len(variants[lab]) > 1:
                lab = " ".join((lab,) + extra[k])
            labeled[lab].append(k)
        if len(labeled) > 1:
            out[gkey] = dict(labeled)
    return out

def compare_sections(tag, keys, out_dir):
    """Part 0: compare_track_modes figures per unique configuration (all tokens except the
    CMP_FAMILY ones) that was run with at least two tracking modes, all its modes overlaid.
    When a configuration was run with several seeds, two seed-averaged figures come first - seeds
    incomplete for some mode dropped from every mode, then modes incomplete for some seed left out
    (see compare_track_modes.make_figure's `incomplete`); w=/r= hashes are ignored for this grouping
    since they usually differ per seed - followed by one figure per seed.
    Groups with more modes than the palette are split into several figures. Returns the html snippets."""
    import compare_track_modes as ctm
    import matplotlib.pyplot as plt
    per_seed = _cmp_entries(keys, set())
    pooled = {g: labs for g, labs in _cmp_entries(keys, {"s"} | set(HASH_TOKENS)).items()
              if len({_split_seed(k)[1] for ks in labs.values() for k in ks}) > 1}
    if not per_seed and not pooled:
        print("No configuration was run with more than one tracking mode - no comparison plots")
        return []
    store = {}
    for p in glob.glob(os.path.join(SCRATCH, f"*_{tag}_s=*_SNR=*.csv")):
        ctm._record(store, os.path.basename(p), lambda p=p: LONG(p))

    # Pooled groups: a (label, seed) pair must map to a single run - otherwise the dropped w=/r=
    # hashes were telling genuinely different runs apart, and averaging them would mix them.
    for gkey in list(pooled):
        for lab, ks in pooled[gkey].items():
            seeds = [_split_seed(k)[1] for k in ks]
            if len(seeds) != len(set(seeds)):
                print(f"  [WARN] no seed-average for [{gkey}]: mode '{lab}' has several runs per seed "
                      f"(they differ in w=/r=, which is not just the seed here)")
                del pooled[gkey]
                break

    # sections: (title, gkey, {label: [keys]}, filename prefix); pooled first, then each seed
    sections = []
    pdiffs = drop_redundant_hashes(differing_tokens(sorted(pooled))) if pooled else {}
    for gi, gkey in enumerate(sorted(pooled, key=lambda g: sort_key(pdiffs[g]))):
        nseeds = len({_split_seed(k)[1] for ks in pooled[gkey].values() for k in ks})
        label = " | ".join(pdiffs[gkey])
        if pdiffs[gkey] == ["config"]:
            label = " | ".join(tok for name, tok in _split_key_tokens(gkey)
                               if name in ("sp", "cfo", "cfod", "R", "aug"))
        sections.append((f"{label} &mdash; average of {nseeds} seeds, incomplete seeds dropped from every mode",
                         gkey, pooled[gkey], f"compare_avg_{gi}", "seeds"))
        sections.append((f"{label} &mdash; average of {nseeds} seeds, only modes complete for every seed",
                         gkey, pooled[gkey], f"compare_avgm_{gi}", "modes"))
    gdiffs = drop_redundant_hashes(differing_tokens(sorted(per_seed))) if per_seed else {}
    for gi, gkey in enumerate(sorted(per_seed, key=lambda g: sort_key(gdiffs[g]))):
        label = " | ".join(gdiffs[gkey])
        if gdiffs[gkey] == ["config"]:   # single configuration on this page - name its main axes instead
            label = " | ".join(tok for name, tok in _split_key_tokens(gkey)
                               if name in ("sp", "cfo", "cfod", "R", "aug"))
        sections.append((label, gkey, per_seed[gkey], f"compare_{gi}", "seeds"))

    html, failed = [], []
    for label, gkey, labeled, prefix, incomplete in sections:
        order = sorted(labeled, key=lambda l: (_trk_group(labeled[l][0]), l))
        seeds = sorted({_split_seed(k)[1] for ks in labeled.values() for k in ks})
        # weights hashes (r=/w=) per seed - the pooled gkey has them stripped (see _cmp_entries)
        hashes = collections.defaultdict(set)
        for ks in labeled.values():
            for k in ks:
                hashes[_split_seed(k)[1]].update(tok for name, tok in _split_key_tokens(k) if name in HASH_TOKENS)
        hash_txt = ", ".join(f"s={sd}: {' '.join(sorted(hashes[sd])) or '-'}" for sd in seeds)
        html.append(f"<h3>{label}</h3><p class='params'>{gkey}"
                    + (f"<br>seeds: {', '.join(map(str, seeds))}<br>weights: {hash_txt}" if len(seeds) > 1 else "")
                    + "</p>")
        # store keys hold the config without its s= token; pass every run behind each label
        chosen_all = {l: tuple(sorted({_split_seed(k)[0] for k in labeled[l]})) for l in order}
        # One per-SNR BLER/MI/BER table per section, with EVERY mode of the section (not split by
        # palette like the figures, so all modes share one seed set): <prefix>.csv, the input of
        # escnn_matlab\master_plot_trk_bler_paper.m (it picks the modes it draws by column name).
        csv_path = os.path.join(out_dir, prefix + ".csv")
        try:
            ctm.write_table(store, chosen_all, order, csv_path, config=gkey,
                            title=label.replace("&mdash;", "-"), seeds=seeds, incomplete=incomplete)
            if os.path.exists(LONG(csv_path)):
                html.append(f"<p class='params'>table (MATLAB): {csv_path}</p>")
        except Exception as e:
            print(f"  [FAIL] table {csv_path}: {e}"); failed.append((label, f"table: {e}"))
        n = len(ctm.COLORS)
        for ci in range(0, len(order), n):
            part = order[ci:ci + n]
            fname = prefix + (f"_{ci // n}" if len(order) > n else "") + ".png"
            print("=" * 70); print(f"[compare {label}]  {' '.join(part)}")
            try:
                chosen = {l: chosen_all[l] for l in part}
                fig = ctm.make_figure(store, chosen, part, label.replace("&mdash;", "-"),
                                      os.path.join(out_dir, fname), seeds=seeds, incomplete=incomplete)
                plt.close(fig)
                html.append(f"<img class='curve' src='{fname}'>")
            except Exception as e:
                print(f"  [FAIL] {e}"); failed.append((label, str(e)))
                html.append(f"<p class='params'>comparison plot failed: {e}</p>")
    print(f"Comparison plots: {len(pooled)} seed-averaged + {len(per_seed)} per-seed configuration(s), "
          f"failed: {len(failed)}")
    return html

def _build_one(tag, name, keys, out_dir=None):
    """Build the analysis page for `keys` (configs of `tag`) into <out_dir>/<name>.html; out_dir
    defaults to ANALYSIS/<name> (rotated if taken - see _rotate_out_dir)."""
    keys = sorted(keys)
    diffs = drop_redundant_hashes(differing_tokens(keys))
    keys.sort(key=lambda k: (_trk_group(k), sort_key(diffs[k])))

    out_dir = out_dir or _rotate_out_dir(name)
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
  span.wts { font-weight: normal; font-family: Consolas, monospace; font-size: 12px; color: #555; }
  p.wtsline { font-family: Consolas, monospace; font-size: 13px; font-weight: bold; }
</style></head><body>""" % name]
    all_hashes = sorted({t for k in keys for t in _hash_tag(k).split()})
    html.append(f"<p class='wtsline'>Weights (SHA1 tags) on this page: {', '.join(all_hashes) or 'none'}</p>")
    cmp_html = compare_sections(tag, keys, out_dir)
    if cmp_html:
        html.append(f"<h1>{name} &mdash; Part 0: tracking-mode comparisons (ESCNN, all modes per configuration)</h1>")
        html += cmp_html
    html.append(f"<h1>{name} &mdash; Part 1: BLER / MI / {third_panel} curves</h1>")

    failed, copied = [], 0
    seen_mat_targets = {}
    for k in keys:
        short = SAFE("_".join(diffs[k]))
        label = " | ".join(display_tokens(k, diffs[k]))
        pattern = f"*_{k}_SNR=*.csv"
        print("=" * 70); print(f"[{label}]  {pattern}")
        has_ue_curve = False
        ue_indices = []
        try:
            aug_type = pmc.detect_aug_type(k) or "LMMSE"
            mat_dir, mat_name, trk_mode = mat_target(name, k, diffs[k], short, aug_type)
            # aug_type alongside trk_mode: the trk-comparison plot also wants
            # the un-augmented baseline (bler_no_aug/mi_no_aug, already saved
            # for every config) drawn as a labeled 5th curve, and needs to
            # know which detector that baseline actually is (LMMSE/SPHERE/...).
            mat_extra = {"trk_mode": trk_mode, "aug_type": aug_type} if trk_mode is not None else None
            if mat_dir is not None:
                mat_key = (mat_dir, mat_name)
                if mat_key in seen_mat_targets:
                    print(f"  [WARN] mat collision: this config's trk={trk_mode} .mat file "
                          f"will overwrite the one just written for config [{seen_mat_targets[mat_key]}] "
                          f"-- they share a trk mode but differ in csg/lr/ep/bs/loss/tw/po, which the bare "
                          f"'{mat_name}.mat' filename can't distinguish.")
                seen_mat_targets[mat_key] = label
            pmc.plot_csvs(pattern, mat_out_dir=mat_dir, mat_name=mat_name, mat_extra=mat_extra,
                          seed_list=[_split_seed(k)[1]])
            if mat_dir is not None:
                print(f"  mat -> {os.path.join(mat_dir, mat_name + '.mat')}  (trk_mode={trk_mode})")
            src = os.path.join(SCRATCH, "plot_output.png")
            if os.path.exists(src):
                shutil.move(src, os.path.join(out_dir, f"curve_{short}.png"))
                print("  [OK] curve")
            else:
                raise RuntimeError("plot_output.png not produced")
            # Second figure with all-UEs-combined lines, plus one figure per
            # individual UE -- all produced only when the CSVs have per-user
            # columns (see plot_multiple_csvs.py).
            src_ue = os.path.join(SCRATCH, "plot_output_peruser.png")
            if os.path.exists(src_ue):
                shutil.move(src_ue, os.path.join(out_dir, f"curve_{short}_peruser.png"))
                has_ue_curve = True
            for src_u in sorted(glob.glob(os.path.join(SCRATCH, "plot_output_ue*.png"))):
                m = re.search(r"plot_output_ue(\d+)\.png$", src_u)
                if not m:
                    continue
                u = int(m.group(1))
                shutil.move(src_u, os.path.join(out_dir, f"curve_{short}_ue{u}.png"))
                ue_indices.append(u)
        except Exception as e:
            print(f"  [FAIL] {e}"); failed.append((label, str(e)))
        plt.close("all")
        html.append(f"{_h3(label, k)}<p class='params'>{k}</p>"
                    f"<img class='curve' src='curve_{short}.png'>")
        if has_ue_curve:
            html.append(f"<img class='curve' src='curve_{short}_peruser.png'>")
        for u in sorted(ue_indices):
            html.append(f"<img class='curve' src='curve_{short}_ue{u}.png'>")

    html.append(f"<h1>{name} &mdash; Part 2: LLR histograms &amp; training loss "
                f"({' | '.join(d.upper() for d in dets)})</h1>")

    def jpg_snrs(k):
        """SNRs for which histogram jpgs exist for this config (any detector)."""
        toks = diffs[k]
        snrs = set()
        for det in dets:
            pat = os.path.join(SCRATCH, f"{det}_2*_{k}_SNR=*.jpg")
            for p in glob.glob(pat):
                m = re.search(r"_SNR=(-?\d+)\.jpg$", os.path.basename(p))
                if m:
                    snrs.add(int(m.group(1)))
            # already-copied short-named jpgs also count (source may be purged)
            for p in glob.glob(os.path.join(out_dir, f"{det}_{SAFE('_'.join(toks))}_SNR*.jpg")):
                m = re.search(r"_SNR(-?\d+)\.jpg$", os.path.basename(p))
                if m:
                    snrs.add(int(m.group(1)))
        return sorted(snrs)

    missing = []
    for k in keys:
        short = SAFE("_".join(diffs[k]))
        label = " | ".join(display_tokens(k, diffs[k]))
        html.append(f"{_h3(label, k)}<p class='params'>{k}</p>")
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
                    pat = os.path.join(SCRATCH, f"{det}_2*_{k}_SNR={snr}.jpg")
                    hits = glob.glob(pat)
                    if hits:
                        shutil.copy2(LONG(hits[0]), LONG(dst)); copied += 1
                if os.path.exists(dst):
                    row.append(f"<td><img src='{sname}'></td>")
                else:
                    row.append("<td>missing</td>"); missing.append((det, label, snr))
            html.append("<tr>" + "".join(row) + "</tr>")
        html.append("</table>")

    html.append("</body></html>")
    out_html = os.path.join(out_dir, f"{name}.html")
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
