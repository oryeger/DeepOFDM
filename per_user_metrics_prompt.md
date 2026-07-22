# Task: Add per-user MI/BER/BLER alongside existing pooled metrics

## Context
`python_code/evaluate.py` currently computes pooled (all-users-combined) MI,
BER, and BLER per SNR/seed and writes them to CSV, which `plot_multiple_csvs.py`
reads and averages across seeds for `curve_<config>.png` plots.

We're debugging a case where a 4-UE Doppler-channel TTA run (`tl=tsyn`) fails
while `tl=bce` is clean, and suspect the pooled statistic hides per-user
variation (users' channels evolve at different rates). We need per-user
MI/BER/BLER visibility WITHOUT changing anything about training, the TTA loss,
or the existing pooled metrics.

## Hard constraints — read carefully
1. **Do not touch the TTA loss computation anywhere.** This is a pure
   post-hoc/reporting change. If you find yourself editing anything upstream
   of metric calculation (e.g. anything under `python_code/detectors/`
   trainers, loss functions), stop — that's out of scope.
2. **The existing pooled columns (`total_mi_1`, `total_ber_1`, `total_bler_...`,
   etc.) must be computed EXACTLY as they are today** — same function calls,
   same arguments, same code path. Do not refactor them, do not derive them
   from the new per-user values (e.g. no averaging 4 per-user values to get
   "overall" — the overall must come from the original unmodified call).
3. **Per-user values must be computed by slicing the SAME tensors that already
   feed the pooled call**, using the same slicing idiom already used elsewhere
   in the file for `conf.ber_on_one_user` (see `evaluate.py` ~line 1230).
   No new forward passes, no new decode calls where avoidable (see BLER note
   below — `crc_out` is already per-user before it gets summed).
4. **No new CLI/config flags for "which user to run."** All 4 users' per-user
   metrics should be computed in the SAME evaluation pass as the pooled ones,
   for every run, not gated behind a config that requires 4 reruns.
5. **`build_analysis.py` should need ZERO changes.** It discovers configs from
   filenames only, never reads CSV columns — the per-user data must ride as
   new columns in the SAME CSV files (same filenames), not new files.

## Where to work

### 1. `python_code/evaluate.py` — MI (ESCNN path)
Around line ~1671 (`calc_mi(tx_data_for_ber.cpu(), llrs_mat_list[iteration].cpu(),
num_bits_data, n_users, num_res)`, and the `calc_mi_from_ldpc` call ~1627):
add a `user_idx: int | None = None` parameter to `calc_mi` and
`calc_mi_from_ldpc` (defined earlier in the same file). When `user_idx` is
given, slice the existing per-user axis (`llr_4[:, user_idx, :]` in `calc_mi`,
`llr_aligned[user_idx]` / `tx_aligned[user_idx]` in `calc_mi_from_ldpc`) BEFORE
flattening, instead of flattening across the user axis. Leave the `None`
(pooled) behavior byte-for-byte identical to today.

At each call site, after the existing unmodified pooled call, add:
```python
for u in range(n_users):
    mi_u = calc_mi(tx_data_for_ber.cpu(), llrs_mat_list[iteration].cpu(),
                    num_bits_data, n_users, num_res, user_idx=u)
    total_mi_user_lists[u][iteration].append(mi_u)
```
(mirror for the `calc_mi_from_ldpc` call site, and for the `mi_lmmse`/`mi_sphere`/
`mi_deeprx`/`mi_e2e` calls if those detectors are in scope — ask me if unsure
which detectors matter, default to at least LMMSE and ESCNN.)

You'll need to initialize `total_mi_user_lists = [[[] for _ in range(iterations)]
for _ in range(n_users)]` near the existing `total_mi_list` initialization
(~line 451).

### 2. `python_code/evaluate.py` — BER
Follow the EXACT slicing idiom already used by the existing
`conf.ber_on_one_user` branches (search for `ber_on_one_user` — appears at
~1230, 1247, 1265, 1281, 1298, 1314, 1330, 1346 for each detector). Example
for the ESCNN path (~1230-1236):
```python
# existing pooled call — DO NOT MODIFY
ber = calculate_ber(detected_word_cur_re.cpu(), target.cpu(), num_bits_data)
ber_sum[iteration] += ber

# new — add per-user loop right after
for u in range(n_users):
    ber_u = calculate_ber(detected_word_cur_re[:, u].unsqueeze(-1).cpu(),
                           target[:, u].unsqueeze(-1).cpu(), num_bits_data)
    ber_sum_user[u][iteration] += ber_u
```
Do this for ESCNN at minimum; replicate for other detectors only if trivial —
flag any detector where the tensor shapes don't obviously support `[:, u]`
slicing rather than guessing.

### 3. `python_code/evaluate.py` — BLER
Around line ~1533-1568. `crc_out = crc.decode(decodedwords)` is ALREADY a
per-user boolean array before `crc_count += (~crc_out).numpy().astype(int).sum()`
collapses it. Do not add any new `codec.decode`/`crc.decode` calls — just also
accumulate per-user inside the same slot loop:
```python
crc_count_per_user = [0] * n_users   # init before the `for slot in range(num_slots)` loop
...
for slot in range(num_slots):
    decodedwords = codec.decode(llr_all_res[:, slot * ldpc_n:(slot + 1) * ldpc_n])
    crc_out = crc.decode(decodedwords)
    crc_count += (~crc_out).numpy().astype(int).sum()          # unchanged
    for u in range(n_users):
        crc_count_per_user[u] += int(~crc_out[u])              # new
...
bler_list[iteration] = crc_count / (num_slots * n_users)        # unchanged
for u in range(n_users):
    bler_user_list[u][iteration] = crc_count_per_user[u] / num_slots
```

### 4. CSV column naming (write, ~line 2044-2072 for the `data_mi`/`to_csv` block,
and the analogous BER/BLER `to_csv` blocks)
Add columns following the SAME naming convention already in use
(`total_ber_1`, `total_ber_deepsic_1`, etc.), just with a `user{u}` token:
- `total_mi_user{u}_{iteration+1}`
- `total_ber_user{u}_{iteration+1}`
- `total_bler_user{u}_{iteration+1}` (in the `_bler.csv` file)

Keep all existing columns unchanged — these are pure additions to the same
DataFrames (`data_mi`, and whatever dict feeds the base/`_bler` CSVs).

### 5. `python_code/utils/plot_multiple_csvs.py`
Follow the EXISTING pattern used for MHSA (search `total_ber_mhsa` — the block
starting ~line 370s in `plot_csvs`). Add an equivalent block that:
- regex-detects `total_{ber,bler,mi}_user(\d+)_(\d+)` columns per SNR file,
- reads them into `seed_snr_dict` the same way the pooled/MHSA columns are,
- averages across seeds the same way,
- plots each user as its own line on the SAME existing BLER/MI/BER axes
  (`axes[subplot_index[...]]`) — use a consistent colormap slot per user index
  (e.g. `plt.cm.tab10(u)`) with a shared linestyle (dotted, matching how ESCNN
  iterations are styled today) so the 4 user lines read as "one detector,
  split 4 ways." The existing pooled ESCNN line (solid, unchanged) stays as
  the "all users together" curve on the same plot — do not remove or restyle it.
- Legend labels: `"ESCNN UE{u}"` per user, alongside the existing `"ESCNN"` label.

`build_analysis.py` should require NO changes — verify this after your edit by
running it against an existing tag and confirming the discovered config set
and file count are identical to before your change.

## Verification (do this before declaring done)
1. Run a small `evaluate.py` sweep (few SNR points, 1 seed is fine) on an
   existing 4-UE tag and diff the pooled columns (`total_mi_1`, `total_ber_1`,
   `total_bler_...`) against a run from before your change (or against `git
   stash` the old version) — they must be numerically identical. This is the
   most important check: if pooled numbers moved at all, something got
   refactored that shouldn't have been.
2. Confirm the new `*_user{u}_*` columns exist in the CSV and that, for a
   single-user (`n_users=1`) run, `total_mi_user0_1 == total_mi_1` exactly
   (sanity check that user-0 slicing matches the pooled/only-user case).
3. Run `build_analysis.py <tag>` on a tag with both old and new CSVs present
   (or a fresh tag) and confirm: same number of `curve_<config>.png` files as
   before, no new config keys appeared, and the per-user lines render on the
   existing plots without errors when the new columns are present, and
   degrade gracefully (just the pooled line, as today) when they're absent —
   e.g. for old CSVs from before this change.
4. Show me the diff of `evaluate.py` and `plot_multiple_csvs.py` before you
   run anything destructive, and call out anywhere you deviated from the plan
   above or made a judgment call (e.g. which non-ESCNN detectors you did or
   didn't add per-user BER for).
