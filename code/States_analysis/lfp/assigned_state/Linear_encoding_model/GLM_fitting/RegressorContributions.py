"""
=============================================================================================
Measure per-regressor contributions to the fitted GLM (step 6a).
=============================================================================================
Pipeline position: step 6a (see README.md). For each channel it reuses the penalty (alphas) and
sample count from the step-4 fit pickle, refits the ridge model to recover the weights (the step-4
pickle stores an UNFITTED model -- cross_val_predict never fits the estimator it is given), and
then quantifies how much each regressor FAMILY contributes:

  * betas        - fitted weights; grouped by regIdx these are each family's per-lag KERNEL shape.
  * trace_var    - variance of each family's contribution trace (its columns @ its weights):
                   "how much this family swings the predicted signal on its own". Fast.
  * dR2          - unique contribution: refit with each family's columns DROPPED (10-fold CV,
                   reusing the full-model alpha) and report the drop in cv-R^2 vs the full fit.
                   Credits a family only for what the others cannot explain. One refit per family.

Writes {designMatID}_contributions.npz per channel. The per-timepoint contribution TRACES are not
stored here (they are n_timepoints x n_families and large); AnalyzeRegressorContributions.py
(step 6b) recomputes them over a short window from betas + the design when plotting.

Channels are parallelised one worker each across SLURM with acme (like DesignMatrix.py /
FittingGLM.py). Lighter than step 4 (the design stays sparse -- no dense alpha search here), but it
does N_families+1 cross-validated ridge fits per channel.

Run in the warping env (needs acme: `conda install -c conda-forge esi-acme`).
"""

import os
import re
import glob
import pickle
import warnings
from pathlib import Path

import numpy as np
from scipy.sparse import diags, load_npz
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score

from acme import ParallelMap

warnings.filterwarnings("ignore", category=FutureWarning)

# -------------------------
# Config
# -------------------------
results_dir = '/cs/projects/MWzeronoise/Analysis/4Shivangi/Results/states_analysis/states_lfp/all_trials/full_length/GLM'

# Which session/channel pairs to process. None -> auto-discover from the results tree (matches
# DesignMatrix.py / FittingGLM.py).
SESSIONS = None   # e.g. ['20230214']
CHANNELS = None   # e.g. [2]; applies to every session. None -> discover per session

# Which fit to read. None -> auto-find the {designMatID}_*samples.pkl (largest sample count if
# several exist), matching ExamineFit.py. Set to pin a specific sample count.
N_TIMEPOINTS = None

# Cross-validation folds for the reduced-model dR2 (match FittingGLM.py's cv=10).
CV_FOLDS = 10

# SLURM / acme parallelisation: one worker per channel. The design stays sparse (no dense alpha
# search), so this is lighter than step 4, but it runs N_families+1 ridge CV fits per channel.
SLURM_PARTITION = '32GBS'
MAX_WORKERS = 100
MEM_PER_WORKER = '32GB'


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


def find_fit(SAVE_PATH, designMatID):
    """Resolve the step-4 fit pickle + its n_timepoints (explicit N_TIMEPOINTS or largest on disk)."""
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


def normalise_design(fullR_sparse):
    """Column-normalise the design exactly as FittingGLM.py does."""
    design_mean = np.array(fullR_sparse.mean(axis=0)).ravel()
    design_sq_mean = np.array(fullR_sparse.power(2).mean(axis=0)).ravel()
    design_std = np.sqrt(design_sq_mean - design_mean**2)
    design_std[design_std == 0] = 1.0
    return fullR_sparse @ diags(1 / design_std)


# -------------------------
# Per-channel worker: full + reduced fits for one channel
# -------------------------
def contributions_channel(channel, session):
    """One acme worker: measure per-family contributions for one channel, write its results/ npz."""
    SESSION_ROOT = Path(results_dir) / session / f'channel{channel}_regressors'
    SAVE_PATH = SESSION_ROOT / "results"
    designMatID = f'{session}_channel{channel}'

    fit_path, n_timepoints = find_fit(SAVE_PATH, designMatID)
    if fit_path is None or not fit_path.exists():
        print(f"skip {designMatID}: no {designMatID}_*samples.pkl (run FittingGLM.py first)")
        return

    # Reuse the step-4 penalty + full-model cv-R^2 so contributions stay consistent with the fit.
    with open(fit_path, 'rb') as f:
        fit = pickle.load(f)
    alphas = fit['alphas']
    full_R2 = fit['scores']

    fullR_sparse = load_npz(os.path.join(SAVE_PATH, f"{designMatID}_dMatProcessed_sparse.npz"))
    meta = np.load(
        os.path.join(SAVE_PATH, f"{designMatID}_dMatProcessed_metadata.npz"),
        allow_pickle=True
    )
    regIdx = np.asarray(meta["regIdx"]).ravel()      # 1-based regressor id per column
    regLabels = np.asarray(meta["regLabels"]).ravel()

    # Load the DOWNSAMPLED neural target written by DesignMatrix.py -- aligned to the design's row
    # rate, exactly as FittingGLM.py fits it. (The raw neural_data.npz is at NATIVE_FS and would be a
    # row-count mismatch with the design.)
    neural_path = os.path.join(SAVE_PATH, f"{designMatID}_neural_downsampled.npz")
    if not os.path.exists(neural_path):
        print(f"skip {designMatID}: no downsampled neural target (re-run DesignMatrix.py first)")
        return
    neural_data = np.load(neural_path)['data']
    if neural_data.ndim == 1:
        neural_data = neural_data[:, None]   # ridge_MML/Ridge expect a 2D (n_samples x n_targets) target

    # Match the step-4 fit window exactly.
    neural_data = neural_data[:n_timepoints]
    fullR_sparse = fullR_sparse[:n_timepoints]
    design_norm = normalise_design(fullR_sparse)

    print(f"{designMatID}: design {design_norm.shape}, full cv-R2 {full_R2:.4f}")

    # --- Full model: fit once to recover the weights (betas) ---
    full_mdl = Ridge(alpha=alphas, fit_intercept=False)
    full_mdl.fit(design_norm, neural_data)
    betas = np.asarray(full_mdl.coef_).ravel()       # single target -> 1D over columns

    # Regressor families (1-based ids), with a readable label each.
    group_ids = np.unique(regIdx)
    group_labels = np.array([regLabels[int(g) - 1] for g in group_ids])

    # --- Per-family trace variance (cheap) and unique dR2 (one reduced refit each) ---
    trace_var = np.full(group_ids.size, np.nan)
    dR2 = np.full(group_ids.size, np.nan)
    for j, g in enumerate(group_ids):
        cols_g = (regIdx == g)

        # contribution trace = this family's columns @ its weights; variance summarises its swing.
        contrib = np.asarray(design_norm[:, cols_g] @ betas[cols_g]).ravel()
        trace_var[j] = float(np.var(contrib))

        # unique contribution: refit without this family, measure the cv-R^2 drop.
        reduced = design_norm[:, ~cols_g]
        reduced_preds = cross_val_predict(
            Ridge(alpha=alphas, fit_intercept=False), reduced, neural_data,
            cv=CV_FOLDS, n_jobs=-1
        )
        reduced_R2 = r2_score(neural_data, reduced_preds)
        dR2[j] = float(full_R2 - reduced_R2)
        print(f"{designMatID}:   {group_labels[j]:<24} dR2={dR2[j]:+.4f}  var={trace_var[j]:.4g}")

    out_file = os.path.join(SAVE_PATH, f"{designMatID}_contributions.npz")
    np.savez_compressed(
        out_file,
        betas=betas,
        regIdx=regIdx,
        regLabels=regLabels,
        alphas=np.asarray(alphas),
        full_R2=full_R2,
        n_timepoints=n_timepoints,
        group_ids=group_ids,
        group_labels=group_labels,
        dR2=dR2,
        trace_var=trace_var,
    )
    print(f"{designMatID}: wrote contributions -> {out_file}")
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

        with ParallelMap(contributions_channel, channels, session,
                         n_inputs=len(channels),
                         partition=SLURM_PARTITION,
                         n_workers=n_workers,
                         mem_per_worker=MEM_PER_WORKER,
                         setup_timeout=600,   # busy cluster: wait up to 10 min for SLURM to allocate
                         write_worker_results=False,   # workers save their own npz; nothing to collect
                         setup_interactive=False) as pmap:
            pmap.compute()
