"""
=============================================================================================
Examine the fitted GLM for every session/channel.
=============================================================================================
Pipeline position: step 5 (see README.md). For every session/channel, loads the fit pickle
written by FittingGLM.py and plots observed vs predicted neural activity over a short window,
saving a PDF into the plots tree (mirroring the results tree but rooted in plots_dir, like
AnalyzeRedundancySubsample.py). Run after the fit (step 4) finishes.

Light / login-node script (no acme): `python ExamineFit.py`.
"""

import os
import re
import glob
import pickle
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)

results_dir = '/cs/projects/MWzeronoise/Analysis/4Shivangi/Results/states_analysis/states_lfp/all_trials/full_length/GLM'
plots_dir = '/cs/projects/MWzeronoise/Analysis/4Shivangi/plots/states_lfp/all_trials/full_length/GLM'

# Which session/channel pairs to examine. None -> auto-discover from the results tree.
SESSIONS = None
CHANNELS = None

# Which fit to load. None -> auto-find the {designMatID}_*samples.pkl in the channel's results/
# folder (use the largest sample count if several exist). Set to pin a specific sample count.
N_TIMEPOINTS = None

# Window (in SECONDS) of the observed/predicted trace to plot. Converted to frames per channel using
# the frame_rate saved with the target, so it stays fixed in real time regardless of DOWNSAMPLE_FACTOR
# (and is clamped to the recording length below).
START_SEC = 100.0
WINDOW_SEC = 15.0
FRAME_RATE_FALLBACK = 100   # Hz, used only if the target npz predates the frame_rate field


def discover_channels(session):
    chans = []
    for p in glob.glob(str(Path(results_dir) / session / 'channel*_regressors')):
        m = re.search(r'channel(\d+)_regressors$', os.path.basename(p))
        if m:
            chans.append(int(m.group(1)))
    return sorted(chans)


def discover_sessions():
    out = []
    for p in sorted(glob.glob(str(Path(results_dir) / '*'))):
        if os.path.isdir(p) and discover_channels(os.path.basename(p)):
            out.append(os.path.basename(p))
    return out


def find_fit(SAVE_PATH, designMatID):
    """Resolve the fit pickle + its n_timepoints (explicit N_TIMEPOINTS or largest on disk)."""
    if N_TIMEPOINTS is not None:
        return SAVE_PATH / f'{designMatID}_{N_TIMEPOINTS}samples.pkl', N_TIMEPOINTS
    candidates = []
    for p in glob.glob(str(SAVE_PATH / f'{designMatID}_*samples.pkl')):
        m = re.search(r'_(\d+)samples\.pkl$', os.path.basename(p))
        if m:
            candidates.append((int(m.group(1)), Path(p)))
    if not candidates:
        return None, None
    n_timepoints, path = max(candidates)   # prefer the largest sample count
    return path, n_timepoints


def examine_channel(session, channel):
    designMatID = f'{session}_channel{channel}'
    SESSION_ROOT = Path(results_dir) / session / f'channel{channel}_regressors'
    SAVE_PATH = SESSION_ROOT / "results"

    fit_path, n_timepoints = find_fit(SAVE_PATH, designMatID)
    if fit_path is None or not fit_path.exists():
        print(f'skip {designMatID}: no {designMatID}_*samples.pkl (run FittingGLM.py first)')
        return

    with open(fit_path, 'rb') as f:
        res = pickle.load(f)
    alphas, mdl, preds, scores = res['alphas'], res['mdl'], res['preds'], res['scores']

    # Load the DOWNSAMPLED neural target (design's row rate), matching FittingGLM.py so it aligns
    # with preds; the raw neural_data.npz is at NATIVE_FS and would be 10x too long for the overlay.
    neural_path = SAVE_PATH / f"{designMatID}_neural_downsampled.npz"
    if not neural_path.exists():
        print(f'skip {designMatID}: no downsampled neural target (re-run DesignMatrix.py first)')
        return
    npz = np.load(neural_path)
    neural_data = npz['data']
    if neural_data.ndim == 1:
        neural_data = neural_data[:, None]
    neural_data_fitted = neural_data[:n_timepoints, :]
    frame_rate = int(npz['frame_rate']) if 'frame_rate' in npz.files else FRAME_RATE_FALLBACK

    print(f'\n===== {designMatID} ({n_timepoints} samples) =====')
    print(f"  R2: {scores:.4f}")
    print(f"  Alpha range: {np.min(alphas):.4f} to {np.max(alphas):.4f}")

    # seconds -> frames at this channel's rate, clamped so we never index past the fitted length
    start_time = min(int(START_SEC * frame_rate), n_timepoints - 1)
    end_time = min(start_time + int(WINDOW_SEC * frame_rate), n_timepoints)

    fig = plt.figure(figsize=(12, 4))
    plt.plot(neural_data_fitted[start_time:end_time], label="Observed", linewidth=2)
    plt.plot(preds[start_time:end_time], label="Predicted", linewidth=2, alpha=0.8)
    plt.xlabel("Frame")
    plt.ylabel("LFP amplitude")
    plt.title(f"{designMatID}: observed vs predicted LFP (R2={scores:.4f})")
    plt.legend()
    plt.tight_layout()

    fig_dir = Path(plots_dir) / session / f'channel{channel}_regressors' / 'results'
    fig_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = fig_dir / f'{designMatID}_{n_timepoints}samples_obs_vs_pred.pdf'
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f'  saved plot -> {pdf_path}')


if __name__ == '__main__':
    sessions = SESSIONS if SESSIONS is not None else discover_sessions()
    if not sessions:
        raise SystemExit(f'No sessions with channel*_regressors found under {results_dir}')
    for session in sessions:
        channels = CHANNELS if CHANNELS is not None else discover_channels(session)
        for channel in channels:
            examine_channel(session, channel)
