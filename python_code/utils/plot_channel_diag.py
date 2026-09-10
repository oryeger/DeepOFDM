"""Plot per-RE channel power from an ekf.py *_diag.h5 diagnostics file (see ekf.py's
save_diag / run_group()). Plots the DMRS-based, noise-free channel estimate ("CE trunc" - what
LMMSE/EKF actually use) by default; pass --noisy to also overlay the noisy version of that same
estimate ("CE noisy"), so a real fade can be told apart from estimation noise at a given
channel_drift_base_index ("cdi"). If the file has it (ekf.py's return_untruncated diagnostic), a
"CE untrunc" curve is always overlaid too - both it and "CE trunc" are noise_var=0, so any gap
between them isolates what the DMRS estimator's delay-domain truncation (L_taps) itself is doing,
which the noisy-vs-trunc comparison can't show (noise is confounded with truncation there, since
both pass through the same truncation). If the file has it (ekf.py's _ground_truth_channel
diagnostic), a "ground truth" curve (thick, dash-dot, drawn first as a background reference) is
also overlaid - a dense, full-resolution estimate that bypasses the DMRS comb/delay-domain
machinery entirely. This is the only curve that's the *actual* channel; the "CE *" curves are all
DMRS-based estimates of it (with different noise/truncation settings) and need to be checked
against ground truth, not just against each other, to tell whether either reconstruction is
actually right.

Each curve type gets its own fixed color (ground truth=black, CE noisy=red, CE trunc=blue, CE
untrunc=orange) so they read apart at a glance; when a file has more than one (ant, user) series,
each additional series within a type is drawn a shade lighter (fading alpha) rather than a new
color, so curve *type* stays the primary visual cue.

Usage:
    python plot_channel_diag.py [--cdi CDI] [--file PATH] [--db] [--noisy] [--clipboard]

With no --file, the newest *_diag.h5 file in Scratchpad is used (ekf.py writes one such file
per run - "newest" is normally the run you just produced). With no --cdi, the first cdi group
in the file is plotted; pass an invalid one to see the full list of what's available. --clipboard
copies the plot as an image to the Windows clipboard (e.g. to paste into OneNote) - Windows only.
"""
import argparse
import glob
import io
import os
import platform
import subprocess
import tempfile

import h5py
import matplotlib.pyplot as plt
import numpy as np

SCRATCHPAD_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "Scratchpad"))


def _long_path(p: str) -> str:
    """Same \\\\?\\ prefix trick ekf.py's own _long_path uses: this repo's filenames (long
    tag=value suffixes, e.g. the embpilots DMRS tags) routinely land the full path just past
    Windows' legacy 260-char MAX_PATH, even though the filename itself is under NTFS's 255-char
    per-component limit - os.path.getmtime/h5py.File (and anything else going through the plain
    Win32 path APIs) fail with WinError 3 on those without this prefix, even though the file
    genuinely exists (ekf.py wrote it successfully using this same prefix)."""
    return ("\\\\?\\" + p) if os.name == 'nt' else p


def _find_latest_diag_file(scratchpad_dir: str) -> str:
    candidates = glob.glob(os.path.join(scratchpad_dir, "*_diag.h5"))
    if not candidates:
        raise FileNotFoundError(f"No *_diag.h5 files found in {scratchpad_dir} "
                                 f"(run ekf.py with conf.snr in conf.save_loss_plot_snr to produce one)")
    return max(candidates, key=lambda p: os.path.getmtime(_long_path(p)))


def _available_cdis(h5file: h5py.File) -> list:
    return sorted(int(name.split("_", 1)[1]) for name in h5file.keys() if name.startswith("cdi_"))


def _copy_fig_to_clipboard(fig):
    """Copy fig to the Windows clipboard as an image (CF_DIB), so it can be pasted directly
    into OneNote/Word/etc. Windows-only - needs pywin32 and Pillow.

    Rendered at 96 DPI (not the higher DPI a saved PNG would normally use) because paste
    targets like OneNote assume 96 DPI for pasted bitmaps and display them at
    pixel_size/96 inches - a higher DPI here would produce a huge on-page image."""
    if platform.system() != "Windows":
        raise RuntimeError("--clipboard is Windows-only (uses win32clipboard)")
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


def plot_channel(file_path: str, cdi: int = None, db: bool = False, show_noisy: bool = False):
    with h5py.File(_long_path(file_path), "r") as f:
        cdis = _available_cdis(f)
        if not cdis:
            raise ValueError(f"{file_path} has no cdi_* groups")
        if cdi is None:
            cdi = cdis[0]
            print(f"[plot_channel_diag] --cdi not given, defaulting to the first available: cdi={cdi}")
        elif cdi not in cdis:
            raise ValueError(f"cdi={cdi} not found in {os.path.basename(file_path)}. Available: {cdis}")

        grp = f[f"cdi_{cdi}"]
        h_abs_true = grp["h_abs_true_per_re"][:]       # (RE, ant, user) - noise-free reference
        h_abs = grp["h_abs_per_re"][:] if show_noisy else None  # (RE, ant, user) - noisy estimate
        h_angle_true = grp["h_angle_true_per_re"][:] if "h_angle_true_per_re" in grp else None
        h_angle = grp["h_angle_per_re"][:] if (show_noisy and "h_angle_per_re" in grp) else None
        # Truncated-vs-untruncated diagnostic (ekf.py's DMRS delay-domain estimator): both are
        # noise_var=0, so any gap between them isolates what the L_taps truncation itself is
        # doing to the estimate - a persistent gap here (unlike noisy-vs-true, which can't tell
        # truncation bias apart from noise) is the actual truncation-bias check. Absent in diag
        # files from before this was added - degrades gracefully.
        has_untrunc = "h_abs_true_untrunc_per_re" in grp
        h_abs_untrunc = grp["h_abs_true_untrunc_per_re"][:] if has_untrunc else None
        h_angle_untrunc = grp["h_angle_true_untrunc_per_re"][:] if has_untrunc else None
        # Independent ground truth (ekf.py's _ground_truth_channel): bypasses the DMRS
        # comb/delay-domain machinery entirely (dense, full-resolution, noise_var=0 per-RE LS) -
        # this is what "true"/"true untrunc" should be compared *against* to tell whether either
        # DMRS reconstruction matches the real channel; comparing them only to each other can only
        # show what the L_taps truncation itself changes, never whether either one is right.
        has_gt = "h_abs_ground_truth_per_re" in grp
        h_abs_gt = grp["h_abs_ground_truth_per_re"][:] if has_gt else None
        h_angle_gt = grp["h_angle_ground_truth_per_re"][:] if has_gt else None
        num_res, n_ants, n_users = h_abs_true.shape

    power_true = h_abs_true.astype(np.float64) ** 2
    power = h_abs.astype(np.float64) ** 2 if show_noisy else None
    power_untrunc = h_abs_untrunc.astype(np.float64) ** 2 if has_untrunc else None
    power_gt = h_abs_gt.astype(np.float64) ** 2 if has_gt else None
    if db:
        power_true = 10 * np.log10(np.maximum(power_true, 1e-20))
        if show_noisy:
            power = 10 * np.log10(np.maximum(power, 1e-20))
        if has_untrunc:
            power_untrunc = 10 * np.log10(np.maximum(power_untrunc, 1e-20))
        if has_gt:
            power_gt = 10 * np.log10(np.maximum(power_gt, 1e-20))

    # Unwrap along the RE (frequency) axis, per ant/user column - the point is to see the
    # channel's phase-vs-frequency trend (e.g. a group-delay-driven ramp/dispersion) without the
    # +-180deg wraparound obscuring it, same motivation as unwrapping a group-delay measurement.
    if h_angle_true is not None:
        angle_true_deg = np.degrees(np.unwrap(h_angle_true.astype(np.float64), axis=0))
    if h_angle is not None:
        angle_deg = np.degrees(np.unwrap(h_angle.astype(np.float64), axis=0))
    if has_untrunc:
        angle_untrunc_deg = np.degrees(np.unwrap(h_angle_untrunc.astype(np.float64), axis=0))
    if has_gt:
        angle_gt_deg = np.degrees(np.unwrap(h_angle_gt.astype(np.float64), axis=0))

    re_axis = np.arange(num_res)
    # Fixed color per curve *type* (not per ant/user series) - the axis of confusion this was
    # rewritten to fix is telling the curve types apart, not telling series apart.
    TYPE_STYLE = {
        "ground_truth": dict(color="black", linestyle="-.", linewidth=2.5, marker=None, label="ground truth"),
        "ce_noisy":     dict(color="tab:red", linestyle="--", marker="x", label="CE noisy"),
        "ce_trunc":     dict(color="tab:blue", linestyle="-", marker="o", label="CE trunc"),
        "ce_untrunc":   dict(color="tab:orange", linestyle=":", marker="s", markersize=4, label="CE untrunc"),
    }

    def _plot_series(ax, data, style_key, ant, user, series_idx, n_series):
        style = dict(TYPE_STYLE[style_key])
        label = style.pop("label")
        if n_series > 1:
            label += f" (ant={ant}, user={user})"
            # Fade later series lighter so curve *type* (the color) stays the dominant cue and
            # multiple series don't all compete for the same fully-opaque color.
            style["alpha"] = max(0.35, 1.0 - 0.2 * series_idx)
        ax.plot(re_axis, data[:, ant, user], **style, label=label)

    ncols = 2 if h_angle_true is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=(9 * ncols, 5))
    ax_power, ax_angle = (axes[0], axes[1]) if ncols == 2 else (axes, None)
    n_series = n_ants * n_users
    series_idx = 0
    for ant in range(n_ants):
        for user in range(n_users):
            if has_gt:
                # Drawn first so it reads as the reference line the other (marked) curves are
                # checked against, rather than just another series.
                _plot_series(ax_power, power_gt, "ground_truth", ant, user, series_idx, n_series)
            _plot_series(ax_power, power_true, "ce_trunc", ant, user, series_idx, n_series)
            if show_noisy:
                _plot_series(ax_power, power, "ce_noisy", ant, user, series_idx, n_series)
            if has_untrunc:
                _plot_series(ax_power, power_untrunc, "ce_untrunc", ant, user, series_idx, n_series)
            if ax_angle is not None:
                if has_gt:
                    _plot_series(ax_angle, angle_gt_deg, "ground_truth", ant, user, series_idx, n_series)
                _plot_series(ax_angle, angle_true_deg, "ce_trunc", ant, user, series_idx, n_series)
                if h_angle is not None:
                    _plot_series(ax_angle, angle_deg, "ce_noisy", ant, user, series_idx, n_series)
                if has_untrunc:
                    _plot_series(ax_angle, angle_untrunc_deg, "ce_untrunc", ant, user, series_idx, n_series)
            series_idx += 1

    ax_power.set_xlabel("RE")
    ax_power.set_ylabel("Power (dB)" if db else "Power")
    ax_power.set_title("Channel power per RE")
    ax_power.grid(True)
    ax_power.legend()

    if ax_angle is not None:
        ax_angle.set_xlabel("RE")
        ax_angle.set_ylabel("angle(H) (deg, unwrapped)")
        ax_angle.set_title("Channel phase per RE")
        ax_angle.grid(True)
        ax_angle.legend()

    fig.suptitle(f"cdi={cdi} - {os.path.basename(file_path)}")
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description="Plot per-RE channel power from an ekf.py *_diag.h5 file.")
    parser.add_argument("--file", type=str, default=None,
                         help="Path to a *_diag.h5 file. Defaults to the newest one in Scratchpad.")
    parser.add_argument("--cdi", type=int, default=None,
                         help="Which channel_drift_base_index (group) to plot. Defaults to the first one in the file.")
    parser.add_argument("--db", action="store_true", help="Plot power in dB instead of linear.")
    parser.add_argument("--noisy", action="store_true",
                         help="Also overlay the noisy calibration-slot estimate (default: true channel only).")
    args = parser.parse_args()

    file_path = args.file or _find_latest_diag_file(SCRATCHPAD_DIR)
    print(f"[plot_channel_diag] using {file_path}")
    fig = plot_channel(file_path, cdi=args.cdi, db=args.db, show_noisy=args.noisy)

    # Always copy to the clipboard (e.g. to paste into OneNote) - a failure here (non-Windows,
    # pywin32/Pillow missing) shouldn't stop the plot itself from showing.
    try:
        _copy_fig_to_clipboard(fig)
        print("[plot_channel_diag] plot copied to clipboard")
    except Exception as e:
        print(f"[plot_channel_diag] could not copy to clipboard: {e}")

    # Hand the image off to the OS viewer (its own process) instead of plt.show(), which would
    # block this terminal until the window is closed.
    try:
        _open_in_viewer(fig)
    except Exception as e:
        print(f"[plot_channel_diag] could not open OS image viewer ({e}); falling back to plt.show()")
        plt.show()


if __name__ == "__main__":
    main()
