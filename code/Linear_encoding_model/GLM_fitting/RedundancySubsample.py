"""
=============================================================================================
Redundancy diagnostic (required step 3a): QR rank check on downsampled subsamples of the design.
=============================================================================================
Pipeline position: step 3a (see README.md). For each channel, repeatedly subsamples its RAW
design matrix in time and QR-checks which regressor groups are linearly redundant, writing one
JSON record per subsample to that channel's `results/{designMatID}_redundancy_{method}.jsonl`.
Summarised by AnalyzeRedundancySubsample.py (step 3b). NOT applied to the fitted matrix -- it
informs the column-drop / regressor choices you make in DesignMatrix.py.

Channels are parallelised one worker each across SLURM with acme (like Regressors.py /
NeuralData.py / DesignMatrix.py): each worker runs the WHOLE factor sweep for its channel and
writes its own jsonl, so there is no cross-channel contention. The factor list / method / repeats
are config constants below (no CLI args).

Run in the warping env (needs acme: `conda install -c conda-forge esi-acme`).
"""

import os
import re
import glob
import math
import random
import warnings
import json
from pathlib import Path

import numpy as np
from scipy.sparse import load_npz

from acme import ParallelMap
warnings.filterwarnings("ignore", category=FutureWarning)

# -------------------------
# Config
# -------------------------
results_dir = '/cs/projects/MWzeronoise/Analysis/4Shivangi/Results/states_analysis/states_lfp/all_trials/full_length/GLM'

# Which session/channel pairs to check. None -> auto-discover from the results tree (matches
# DesignMatrix.py).
SESSIONS = None   # e.g. ['20230214']
CHANNELS = None   # e.g. [2]; applies to every session. None -> discover per session

# Subsampling sweep. The raw matrix is already at NATIVE_FS/DOWNSAMPLE (e.g. 100 Hz), so keep the
# factors small -- the MIN_ROWS_PER_COL_RATIO guard auto-skips any that would thin rows below the 
# column count (see below).
DOWNSCALE_FACTORS = (1, 2, 4, 8, 16, 32, 64, 128)
DOWNSCALE_METHOD = 'random'   # 'uniform' (start::factor stride) | 'random' (random row subset)
REPEATS = 0.3                 # subsamples per factor = ceil(REPEATS * factor)

# Self-protection against artifactual redundancy. Row-subsampling can only LOWER rank, so once the
# kept rows fall near/below the column count the QR reports the design as rank-deficient purely
# because the matrix is wide -- not because of real collinearity. We therefore skip any factor that
# would leave fewer than MIN_ROWS_PER_COL_RATIO * n_cols rows. This makes the factor sweep safe at
# any sampling rate: the big factors that were fine on a 1 kHz matrix get auto-skipped here instead
# of logging garbage.
MIN_ROWS_PER_COL_RATIO = 3

# SLURM / acme parallelisation: one worker per channel. Each worker densifies subsampled matrices;
# the heavy point is factor=1 (densifies the FULL raw matrix + QR workspace), which at 100 Hz can
# reach the low tens of GB. Bump MEM_PER_WORKER / partition tier if a worker OOMs, or drop factor 1
# from DOWNSCALE_FACTORS (its exact full-matrix verdict is already printed by DesignMatrix.py).
SLURM_PARTITION = '96GBS'
MAX_WORKERS = 100
MEM_PER_WORKER = '96GB'

# Validate config early
if not (0 < REPEATS <= 1):
    raise ValueError("REPEATS must be in the range (0, 1].")
if DOWNSCALE_METHOD not in ('uniform', 'random'):
    raise ValueError("DOWNSCALE_METHOD must be 'uniform' or 'random'.")
if any(f <= 0 for f in DOWNSCALE_FACTORS):
    raise ValueError("DOWNSCALE_FACTORS must be positive integers.")


def discover_channels(session):
    """Channel numbers with a channel<ch>_regressors folder under this session."""
    chans = []
    for p in glob.glob(str(Path(results_dir) / session / 'channel*_regressors')):
        m = re.search(r'channel(\d+)_regressors$', os.path.basename(p))
        if m:
            chans.append(int(m.group(1)))
    return sorted(chans)


def discover_sessions():
    """Session folders that contain at least one channel*_regressors folder."""
    out = []
    for p in sorted(glob.glob(str(Path(results_dir) / '*'))):
        if os.path.isdir(p) and discover_channels(os.path.basename(p)):
            out.append(os.path.basename(p))
    return out


# Define fn to calculate redundancy in the design matrix
def CalculateRedundancy(fullR, regIdx, regLabels):
    # Remove sparse columns by summing absolute values and applying a threshold
    rejIdx = np.nansum(np.abs(fullR), axis=0) < 10
    Rkeep = fullR[:, ~rejIdx]

    # Compute column norms
    col_norm = np.sqrt(np.sum(Rkeep**2, axis=0))
    col_norm[col_norm == 0] = 1.0

    # Normalize columns
    X = Rkeep / col_norm

    # Make sure the matrix is contiguous in memory for efficient computations
    X = np.ascontiguousarray(X)

    # Perform QR decomposition to orthogonalize the design matrix
    _, fullQRR = np.linalg.qr(X, mode="reduced")

    diagR = np.abs(np.diag(fullQRR))
    r1 = np.asarray(fullQRR).ravel(order="F")[0]
    threshold = max(fullR.shape) * np.spacing(abs(r1))
    rank_mask = diagR > threshold

    if np.sum(rank_mask) < fullQRR.shape[1]:
        redundant_cols = ~rank_mask
        good_idx = np.where(~rejIdx)[0]
        rejIdx[good_idx] = redundant_cols

    fullR = fullR[:, ~rejIdx]
    regIdx_kept = regIdx[~rejIdx]

    diag_vals = np.abs(np.diag(fullQRR))
    threshold = max(fullR.shape) * np.spacing(abs(fullQRR.flat[0]))
    zero_cols = np.where(diag_vals <= threshold)[0]
    zero_regs = regIdx[zero_cols]
    unique_regs = np.unique(zero_regs)

    if unique_regs.size == 0:
        redundant_regs = []
    else:
        redundant_regs = [
            regLabels[int(r) - 1]
            for r in unique_regs
        ]

    return np.where(rejIdx), redundant_regs


# -------------------------
# Per-channel worker: full factor sweep for one channel
# -------------------------
def run_channel_sweep(channel, session):
    """One acme worker: sweep all DOWNSCALE_FACTORS for one channel, write its results/ jsonl."""
    SESSION_ROOT = Path(results_dir) / session / f'channel{channel}_regressors'
    SAVE_PATH = SESSION_ROOT / "results"
    designMatID = f'{session}_channel{channel}'

    raw_path = os.path.join(SAVE_PATH, f"{designMatID}_dMatRaw_sparse.npz")
    if not os.path.exists(raw_path):
        print(f"skip {designMatID}: no raw design matrix (run DesignMatrix.py first)")
        return

    fullR_sparse = load_npz(raw_path)
    meta = np.load(
        os.path.join(SAVE_PATH, f"{designMatID}_dMatRaw_metadata.npz"),
        allow_pickle=True
    )
    regIdx = meta["regIdx"]
    regLabels = meta["regLabels"]

    n_t, n_cols = fullR_sparse.shape
    min_rows = MIN_ROWS_PER_COL_RATIO * n_cols

    records = []
    for downscale_factor in DOWNSCALE_FACTORS:
        # Guard: skip factors that would thin the rows below MIN_ROWS_PER_COL_RATIO * n_cols, where
        # the QR would report false (shape-driven) redundancy instead of a real verdict.
        kept_rows = n_t // downscale_factor
        if kept_rows < min_rows:
            print(f"{designMatID}: skip factor {downscale_factor} -> would leave {kept_rows} rows "
                  f"< {MIN_ROWS_PER_COL_RATIO}x{n_cols} cols (row-subsampling below n_cols "
                  f"manufactures false redundancy)")
            continue

        print(f"{designMatID}: {DOWNSCALE_METHOD} subsampling, factor {downscale_factor}")
        for i in range(math.ceil(REPEATS * downscale_factor)):
            if DOWNSCALE_METHOD == 'uniform':
                start = random.randrange(downscale_factor)
                fullR = fullR_sparse[start::downscale_factor, :].toarray()
                extra = {"start": start}
            else:  # 'random'
                random_rows = np.random.choice(n_t, n_t // downscale_factor, replace=False)
                fullR = fullR_sparse[random_rows, :].toarray()
                extra = {"random_rows": random_rows.tolist()}

            rejIdx, redundant_regs = CalculateRedundancy(fullR, regIdx, regLabels)
            records.append({
                "session": session,
                "channel": channel,
                "downscale_factor": downscale_factor,
                "downscale_method": DOWNSCALE_METHOD,
                "repeats": REPEATS,
                "iteration": i,
                "rejIdx": rejIdx[0].tolist(),
                "redundant_regs": redundant_regs,
                **extra,
            })

    # Write once (truncate) so reruns don't accumulate stale records -- no external cleanup needed.
    out_file = os.path.join(SAVE_PATH, f"{designMatID}_redundancy_{DOWNSCALE_METHOD}.jsonl")
    with open(out_file, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"{designMatID}: wrote {len(records)} records -> {out_file}")
    return out_file


# -------------------------
# Driver: parallelise channels across the cluster (one worker per channel, per session)
# -------------------------
if __name__ == '__main__':
    sessions = SESSIONS if SESSIONS is not None else discover_sessions()
    if not sessions:
        raise SystemExit(f'No sessions with channel*_regressors found under {results_dir}')

    for session in sessions:
        channels = CHANNELS if CHANNELS is not None else discover_channels(session)
        if not channels:
            print(f'[{session}] no channel*_regressors folders, skipping')
            continue

        n_workers = min(MAX_WORKERS, len(channels))
        print(f"[{session}] {len(channels)} channels -> {n_workers} workers on '{SLURM_PARTITION}'")

        with ParallelMap(run_channel_sweep, channels, session,
                         n_inputs=len(channels),
                         partition=SLURM_PARTITION,
                         n_workers=n_workers,
                         mem_per_worker=MEM_PER_WORKER,
                         setup_timeout=600,   # busy cluster: wait up to 10 min for SLURM to allocate
                         write_worker_results=False,   # workers write their own jsonl; nothing to collect
                         setup_interactive=False) as pmap:
            pmap.compute()
