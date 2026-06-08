"""
Load and plot U'n'Eye saccade detection results for one session.

Data location:
    /cs/projects/MWzeronoise/Analysis/4Shivangi/Results/saccade_detection/stitched_sessions.npz

Per session fields (keyed as "<session>__<field>"):
    x_nan, y_nan         - eye position (deg), NaN at missing samples
    pred_nan             - binary saccade prediction (0/1), NaN at missing
    prob_sacc_nan        - saccade probability [0,1], NaN at missing
    x_orig, y_orig       - eye position with interpolated NaNs (no gaps)
    pred_orig            - saccade prediction (no gaps)
    prob_sacc_orig       - saccade probability (no gaps)
    nan_mask             - True where original signal had NaN

Sessions available:
    20220825, 20230202, 20230203, 20230206, 20230207,
    20230208, 20230209, 20230213, 20230214

Sampling rate: 500 Hz

Usage in other scripts:
    from load_saccade_data import load_session
    x, y, pred, prob, fs = load_session("20230206")
    onsets = np.where(np.diff(np.nan_to_num(pred)) == 1)[0] + 1
"""

import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = "/cs/projects/MWzeronoise/Analysis/4Shivangi/Results/saccade_detection/stitched_sessions.npz"


def load_session(session):
    """Load one session. Returns (x, y, pred, prob, fs)."""
    d = np.load(DATA_PATH, allow_pickle=True)
    fs = int(d["fs"])
    session = str(session).strip()
    x = d[f"{session}__x_nan"]
    y = d[f"{session}__y_nan"]
    pred = d[f"{session}__pred_nan"]
    prob = d[f"{session}__prob_sacc_nan"]
    return x, y, pred, prob, fs


if __name__ == "__main__":
    session = "20230214"
    plot_dir = "/mnt/hpc/projects/MWzeronoise/Analysis/4Shivangi/plots/saccades"

    x, y, pred, prob, fs = load_session(session)
    time = np.arange(len(x)) / fs

    sacc_idx = np.where(np.diff(np.nan_to_num(pred)) == 1)[0]
    n_sacc = len(sacc_idx)
    dur_s = len(x) / fs
    pct_nan = np.isnan(x).mean() * 100
    print(f"Session {session}: {dur_s:.1f}s, {n_sacc} saccades, {pct_nan:.1f}% NaN")

    # --- Full session overview ---
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(f"Session {session} — full overview", fontsize=14)

    axes[0].plot(time, x, lw=0.3, color="steelblue")
    axes[0].set_ylabel("X position (deg)")

    axes[1].plot(time, y, lw=0.3, color="darkorange")
    axes[1].set_ylabel("Y position (deg)")

    axes[2].plot(time, pred, lw=0.3, color="green", alpha=0.7)
    axes[2].set_ylabel("Saccade prediction")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_yticks([0, 1])

    plt.tight_layout()
    fpath = f"{plot_dir}/{session}_overview.png"
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fpath}")

    # --- Zoomed 10s snippet around an early saccade ---
    center = sacc_idx[5] if len(sacc_idx) > 5 else len(x) // 2
    t0 = max(0, center - 5 * fs)
    t1 = min(len(x), center + 5 * fs)
    sl = slice(int(t0), int(t1))

    fig, axes = plt.subplots(3, 1, figsize=(14, 6), sharex=True)
    fig.suptitle(f"Session {session} — zoomed ({time[int(t0)]:.1f}–{time[int(t1)-1]:.1f} s)", fontsize=14)

    axes[0].plot(time[sl], x[sl], lw=0.8, color="steelblue")
    axes[0].set_ylabel("X (deg)")

    axes[1].plot(time[sl], y[sl], lw=0.8, color="darkorange")
    axes[1].set_ylabel("Y (deg)")

    axes[2].plot(time[sl], prob[sl], lw=0.8, color="gray", alpha=0.5, label="P(saccade)")
    axes[2].fill_between(time[sl], 0, pred[sl], color="green", alpha=0.3, label="Prediction")
    axes[2].axhline(0.5, ls="--", color="red", alpha=0.4, lw=0.8)
    axes[2].set_ylabel("Saccade prob / pred")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend(loc="upper right")

    plt.tight_layout()
    fpath = f"{plot_dir}/{session}_zoomed.png"
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fpath}")
