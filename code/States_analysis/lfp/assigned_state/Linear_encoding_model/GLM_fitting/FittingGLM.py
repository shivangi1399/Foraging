"""
=============================================================================================
Fit the ridge-regression encoding model for every session/channel.
=============================================================================================
Pipeline position: step 4 (see README.md). For each channel, loads the processed design matrix
({designMatID}_dMatProcessed_*) and neural_data.npz, column-normalises the design, picks the
ridge penalty per target with ridge_MML, fits Ridge(fit_intercept=False) with 10-fold CV, and
saves R^2 + model/preds to {designMatID}_{n_timepoints}samples.pkl.

Runs over MANY session/channel pairs: by default it auto-discovers every
`<session>/channel<ch>_regressors` folder under `results_dir`; set SESSIONS / CHANNELS to pin a
subset. All the per-channel work lives in `fit_channel()`, which is parallelised one worker per
channel across SLURM with acme (like DesignMatrix.py / RedundancySubsample.py / Regressors.py).

Heavy (large dense design matrix, sklearn n_jobs=-1) -> each worker needs lots of memory + cores;
MEM_PER_WORKER / SLURM_PARTITION below carry over the old FittingGLM.sh request (~500GB, 24 cores,
10h on a big-mem partition). Tune them / MAX_WORKERS to taste.

Run in the warping env (needs acme: `conda install -c conda-forge esi-acme`).
"""

import os
import re
import glob
import warnings
import pickle
from pathlib import Path

import numpy as np
from scipy.sparse import diags, load_npz
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score

from acme import ParallelMap
from reg import ridge_MML

warnings.filterwarnings("ignore", category=FutureWarning)

# -------------------------
# Config
# -------------------------
results_dir = '/cs/projects/MWzeronoise/Analysis/4Shivangi/Results/states_analysis/states_lfp/all_trials/full_length/GLM'

# Which session/channel pairs to fit. None -> auto-discover from the results tree (matches
# DesignMatrix.py / RedundancySubsample.py).
SESSIONS = None   # e.g. ['20230214']
CHANNELS = None   # e.g. [2]; applies to every session. None -> discover per session

# How many leading time points to fit. None -> the full recording (neural_data.shape[0]).
N_TIMEPOINTS = None   # e.g. 1000000

# SLURM / acme parallelisation: one worker per channel. Each worker densifies the full processed
# design matrix and runs ridge CV with sklearn n_jobs=-1, so it is memory- and CPU-heavy. On a 
# per-core-mem partition acme turns MEM_PER_WORKER into enough cores to feed n_jobs=-1. 
SLURM_PARTITION = '96GB'
MAX_WORKERS = 100
MEM_PER_WORKER = '500GB'


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


# -------------------------
# Per-channel worker: fit one channel
# -------------------------
def fit_channel(channel, session):
    """One acme worker: fit the ridge encoding model for one channel, write its results/ pkl."""
    SESSION_ROOT = Path(results_dir) / session / f'channel{channel}_regressors'
    SAVE_PATH = SESSION_ROOT / "results"
    designMatID = f'{session}_channel{channel}'

    proc_path = os.path.join(SAVE_PATH, f"{designMatID}_dMatProcessed_sparse.npz")
    if not os.path.exists(proc_path):
        print(f"skip {designMatID}: no processed design matrix (run DesignMatrix.py first)")
        return

    # Load in the processed design matrix
    fullR_sparse = load_npz(proc_path)

    meta = np.load(
        os.path.join(SAVE_PATH, f"{designMatID}_dMatProcessed_metadata.npz"),
        allow_pickle=True
    )

    regIdx = meta["regIdx"]
    regLabels = meta["regLabels"]

    # Load in the neural data
    neural_data = np.load(os.path.join(SESSION_ROOT, "neural_data.npz"))['data']

    n_timepoints = neural_data.shape[0] if N_TIMEPOINTS is None else N_TIMEPOINTS

    # Filter the neural data and design matrix to the number of time points we are using
    neural_data = neural_data[:n_timepoints]
    fullR_sparse = fullR_sparse[:n_timepoints]

    print(f"{designMatID}: neural_data shape:", neural_data.shape)
    print(f"{designMatID}: fullR_sparse shape:", fullR_sparse.shape)

    # Fitting the linear model
    design_mean = np.array(fullR_sparse.mean(axis=0)).ravel()
    design_sq_mean = np.array(fullR_sparse.power(2).mean(axis=0)).ravel()
    design_std = np.sqrt(design_sq_mean - design_mean**2)
    design_std[design_std == 0] = 1.0
    design_norm = fullR_sparse @ diags(1 / design_std)

    alphas = ridge_MML(neural_data, design_norm.toarray(), regress=False)

    mdl = Ridge(alpha=alphas, fit_intercept=False)
    preds = cross_val_predict(mdl, design_norm, neural_data, cv=10, n_jobs=-1)
    scores = r2_score(neural_data, preds)

    print(f"{designMatID}: R2: {scores:.4f}")
    print(f"{designMatID}: Alpha range: {np.min(alphas):.4f} to {np.max(alphas):.4f}")

    # Save the fitting results
    fitting_results = {
        'alphas': alphas,
        'mdl': mdl,
        'preds': preds,
        'scores': scores,
    }

    filename = f'{designMatID}_{n_timepoints}samples.pkl'
    out_file = os.path.join(SAVE_PATH, filename)
    with open(out_file, 'wb') as f:
        pickle.dump(fitting_results, f)
    print(f"{designMatID}: wrote results -> {out_file}")
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

        with ParallelMap(fit_channel, channels, session,
                         n_inputs=len(channels),
                         partition=SLURM_PARTITION,
                         n_workers=n_workers,
                         mem_per_worker=MEM_PER_WORKER,
                         write_worker_results=False,   # workers save their own pkl; nothing to collect
                         setup_interactive=False) as pmap:
            pmap.compute()
