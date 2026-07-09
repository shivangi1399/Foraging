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
                   NOTE: the per-trial state levels (state_0, state_2, ...) are mutually-exclusive
                   levels of ONE categorical factor, so they are MERGED into a single "state" family
                   and dropped together -- otherwise each level is redundant with the others (+
                   trial_onset) and its unique dR2 collapses to ~0 (a design artifact, not absence
                   of state encoding).
  * beta_sig /   - per-lag significance: a circular-shift permutation null (shift the target, refit
    beta_thr /     the full ridge, repeat N_PERM times) gives, per family, a max-statistic threshold
    family_p       across its lags (FWER-controlled). beta_sig marks the design columns whose |beta|
                   beats it; family_p is the family-level max-stat permutation p-value. Because the
                   null refits the FULL model, this is "unique given the other regressors" -- a lag
                   can be non-significant here yet clear in the marginal STA/ETA (e.g. target_in_RF
                   vs stim_onset). Set N_PERM=0 to skip.

Writes {designMatID}_contributions.npz per channel. The per-timepoint contribution TRACES are not
stored here (they are n_timepoints x n_families and large); AnalyzeRegressorContributions.py
(step 6b) recomputes them over a short window from betas + the design when plotting.

Channels are parallelised one worker each across SLURM with acme (like DesignMatrix.py /
FittingGLM.py). Lighter than step 4 (the design stays sparse -- no dense alpha search here), but it
does N_families+1 cross-validated ridge fits per channel.

Run in the warping env (needs acme: `conda install -c conda-forge esi-acme`).
"""

import os
import sys
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
# make glm_config (single source of truth for the output tree + sampling rate) importable
for _d in (os.path.dirname(os.path.abspath(__file__)),
           os.path.dirname(os.path.dirname(os.path.abspath(__file__)))):
    if os.path.exists(os.path.join(_d, 'glm_config.py')):
        sys.path.insert(0, _d)
        break
from glm_config import RESULTS_DIR
results_dir = RESULTS_DIR

# Which session/channel pairs to process. None -> auto-discover from the results tree (matches
# DesignMatrix.py / FittingGLM.py).
SESSIONS = None   # e.g. ['20230214']
CHANNELS = None   # e.g. [2]; applies to every session. None -> discover per session

# Which fit to read. None -> auto-find the {designMatID}_*samples.pkl (largest sample count if
# several exist), matching ExamineFit.py. Set to pin a specific sample count.
N_TIMEPOINTS = None

# Cross-validation folds for the reduced-model dR2 (match FittingGLM.py's cv=10).
CV_FOLDS = 10

# Per-lag significance via a circular-shift permutation null (set N_PERM=0 to skip). Each
# permutation circularly shifts the neural target by a random offset >= PERM_MIN_SHIFT_S seconds
# (>> the longest kernel, so the event->LFP link is broken while each signal's autocorrelation is
# preserved) and refits the full ridge ONCE; per family we FWER-correct across its lags with a
# max-statistic (as in the saccade STA / RF-entry ETA). Costs N_PERM extra single ridge fits per
# channel (no CV) on top of the dR2 work -- budget the partition time accordingly.
N_PERM = 1000
ALPHA = 0.05
PERM_MIN_SHIFT_S = 10.0    # min circular shift (s); must exceed the longest kernel window
PERM_SEED = 0
# Save the FULL per-column (per-lag) null betas, not just the per-family max. Needed so the POOLED
# mean-kernel test (AnalyzePooledKernels.py, step 6c) can pool per-lag nulls across channels by the
# shared shift index. Costs an extra (n_columns x N_PERM) float32 array per channel on disk (a few
# MB); set False to keep only the max-stat fam_maxnull (array-level test still works, but the pooled
# per-lag kernel test does not).
SAVE_COL_NULL = True

# SLURM / acme parallelisation: one worker per channel. The design stays sparse (no dense alpha
# search), so this is lighter than step 4, but it runs N_families+1 ridge CV fits per channel
# plus N_PERM single ridge fits for the permutation significance.
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


def session_blocked_folds(session_id, n_folds):
    """CV folds that never straddle a session boundary (POOLED fits only): within each session split
    the contiguous rows into n_folds blocks; fold k = block-k pooled across sessions. See FittingGLM."""
    session_id = np.asarray(session_id).ravel()
    idx = np.arange(session_id.size)
    test_parts = [[] for _ in range(n_folds)]
    for s in np.unique(session_id):
        rows = idx[session_id == s]
        for k, block in enumerate(np.array_split(rows, n_folds)):
            test_parts[k].append(block)
    folds = []
    for k in range(n_folds):
        test = np.concatenate(test_parts[k]) if test_parts[k] else np.array([], dtype=int)
        folds.append((np.setdiff1d(idx, test), test))
    return folds


def build_shift_schedule(n_rows, session_id, min_shift, n_perm, seed):
    """Deterministic per-permutation circular shifts, generated ONLY from (seed, row layout) so they
    are IDENTICAL across every channel that shares the same layout (all channels of a session, or all
    pooled electrodes over the same sessions have the same n_rows / session lengths). That makes
    "permutation p" the SAME shuffle on every channel -- and, crucially, shifting all channels by the
    same offset keeps them mutually aligned, so each shuffle PRESERVES the cross-channel correlation.
    That is what lets an array-level test pool the per-channel nulls by permutation index and stay
    well-calibrated (not anti-conservative).

    Returns (roll(y, p) -> shifted target, shifts) where `shifts` is the stored schedule:
      * single session: shape (n_perm,)      -- one offset per permutation
      * pooled:         shape (n_sess, n_perm) -- per-session offsets (rows ordered by unique session id)
    """
    rng = np.random.default_rng(seed)
    if session_id is None:
        shifts = rng.integers(min_shift, n_rows - min_shift, size=n_perm)

        def roll(y, p):
            return np.roll(y, int(shifts[p]), axis=0)
        return roll, shifts

    sess = np.unique(session_id)
    rows_by_s = [np.where(session_id == s)[0] for s in sess]
    sched = np.zeros((len(sess), n_perm), dtype=int)
    for i, rows in enumerate(rows_by_s):
        n = rows.size
        lo = min(min_shift, max(1, n // 4))
        hi = max(lo + 1, n - lo)
        sched[i] = rng.integers(lo, hi, size=n_perm)

    def roll(y, p):
        out = y.copy()
        for i, rows in enumerate(rows_by_s):
            out[rows] = np.roll(y[rows], int(sched[i, p]), axis=0)
        return out
    return roll, sched


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
    # POOLED (multi-session) designs carry a per-row session_id (from AssemblePooled.py): drives
    # session-aware CV folds + within-session permutation below. Absent -> single-session, unchanged.
    session_id = np.asarray(meta['session_id']).ravel() if 'session_id' in meta.files else None

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
    if session_id is not None:
        session_id = session_id[:n_timepoints]
    cv = session_blocked_folds(session_id, CV_FOLDS) if session_id is not None else CV_FOLDS
    design_norm = normalise_design(fullR_sparse)

    print(f"{designMatID}: design {design_norm.shape}, full cv-R2 {full_R2:.4f}")

    # --- Full model: fit once to recover the weights (betas) ---
    full_mdl = Ridge(alpha=alphas, fit_intercept=False)
    full_mdl.fit(design_norm, neural_data)
    betas = np.asarray(full_mdl.coef_).ravel()       # single target -> 1D over columns

    # Families to score. Normally one per regIdx id, BUT the per-trial state levels
    # (state_0, state_2, ...) are mutually-exclusive levels of ONE categorical factor: with the
    # near-empty levels dropped in DesignMatrix.py they nearly partition every trial, so scored
    # separately each state is redundant with the others (+ trial_onset) and its unique dR2
    # collapses to ~0 (even dipping below the dummy null). We MERGE all state_* levels into a single
    # "state" family so the reduced-model drop removes them together and dR2 reports the honest
    # unique contribution of behavioural state as a factor. (family id = -1 marks the merged state.)
    group_ids = np.unique(regIdx)
    families = []                                    # (label, family-id, column-mask over design)
    state_cols = np.zeros(regIdx.shape, dtype=bool)
    for g in group_ids:
        lab = str(regLabels[int(g) - 1])
        if lab.startswith('state_'):
            state_cols |= (regIdx == g)
        else:
            families.append((lab, int(g), regIdx == g))
    if state_cols.any():
        families.append(('state', -1, state_cols))

    group_labels = np.array([lab for lab, _, _ in families])
    group_family_ids = np.array([gid for _, gid, _ in families], dtype=int)

    # --- Per-family trace variance (cheap) and unique dR2 (one reduced refit each) ---
    trace_var = np.full(len(families), np.nan)
    dR2 = np.full(len(families), np.nan)
    for j, (lab, _gid, cols_g) in enumerate(families):
        # contribution trace = this family's columns @ its weights; variance summarises its swing.
        contrib = np.asarray(design_norm[:, cols_g] @ betas[cols_g]).ravel()
        trace_var[j] = float(np.var(contrib))

        # unique contribution: refit without this family, measure the cv-R^2 drop.
        reduced = design_norm[:, ~cols_g]
        reduced_preds = cross_val_predict(
            Ridge(alpha=alphas, fit_intercept=False), reduced, neural_data,
            cv=cv, n_jobs=-1
        )
        reduced_R2 = r2_score(neural_data, reduced_preds)
        dR2[j] = float(full_R2 - reduced_R2)
        print(f"{designMatID}:   {lab:<24} dR2={dR2[j]:+.4f}  var={trace_var[j]:.4g}")

    # --- Per-lag significance: circular-shift permutation null, per-family max-stat FWER ---
    # Shift the target by a large random offset (breaks event->LFP link, keeps autocorrelation),
    # refit the full ridge, and per family collect max|null beta| over its lags. A lag is
    # significant if |real beta| beats the (1-ALPHA) quantile of that family's max-null; family_p is
    # the max-stat permutation p-value. One refit per permutation covers every family/lag at once.
    beta_thr = np.full(len(families), np.nan)      # per family: (1-ALPHA) quantile of max|null beta|
    beta_sig = np.zeros(betas.shape, dtype=bool)   # per design column
    family_p = np.full(len(families), np.nan)      # per family: max-stat permutation p-value
    real_max = np.full(len(families), np.nan)      # per family: max|real beta| (the real max-statistic)
    fam_maxnull = np.zeros((len(families), N_PERM)) if N_PERM > 0 else np.zeros((len(families), 0))
    # full per-column (per-lag) SIGNED null betas: (n_columns x N_PERM). Kept (when SAVE_COL_NULL)
    # so the pooled mean-kernel test can average per-lag nulls across channels by shift index.
    col_null = (np.zeros((betas.shape[0], N_PERM), dtype=np.float32)
                if (N_PERM > 0 and SAVE_COL_NULL) else np.zeros((betas.shape[0], 0), dtype=np.float32))
    perm_shifts = np.zeros(0)
    if N_PERM > 0:
        frame_rate = int(meta['frame_rate']) if 'frame_rate' in meta.files else 100
        n_rows = design_norm.shape[0]
        min_shift = min(max(int(PERM_MIN_SHIFT_S * frame_rate), 1), max(1, n_rows // 4))
        # deterministic shift schedule -> IDENTICAL across all channels of this array (same seed + same
        # row layout), so permutation p is the same shuffle everywhere and the array-level test can
        # pool the per-channel nulls by permutation index (see build_shift_schedule).
        roll, perm_shifts = build_shift_schedule(n_rows, session_id, min_shift, N_PERM, PERM_SEED)
        perm_mdl = Ridge(alpha=alphas, fit_intercept=False)
        print(f"{designMatID}: running {N_PERM} circular-shift permutations "
              f"(min shift {min_shift} frames, seed {PERM_SEED}) ...")
        for p in range(N_PERM):
            perm_mdl.fit(design_norm, roll(neural_data, p))
            b_null = np.asarray(perm_mdl.coef_).ravel()
            if SAVE_COL_NULL:
                col_null[:, p] = b_null.astype(np.float32)   # keep the full per-lag null for step 6c
            for j, (_lab, _gid, cols_g) in enumerate(families):
                fam_maxnull[j, p] = np.max(np.abs(b_null[cols_g])) if cols_g.any() else 0.0
        for j, (lab, _gid, cols_g) in enumerate(families):
            thr = float(np.percentile(fam_maxnull[j], 100 * (1 - ALPHA)))
            beta_thr[j] = thr
            beta_sig[cols_g] = np.abs(betas[cols_g]) > thr
            real_max[j] = float(np.max(np.abs(betas[cols_g]))) if cols_g.any() else 0.0
            family_p[j] = float(np.mean(fam_maxnull[j] >= real_max[j]))
            print(f"{designMatID}:   {lab:<24} thr={thr:.4g}  "
                  f"{int(beta_sig[cols_g].sum())}/{int(cols_g.sum())} lags sig  p={family_p[j]:.3g}")

    out_file = os.path.join(SAVE_PATH, f"{designMatID}_contributions.npz")
    np.savez_compressed(
        out_file,
        betas=betas,
        regIdx=regIdx,
        regLabels=regLabels,
        alphas=np.asarray(alphas),
        full_R2=full_R2,
        n_timepoints=n_timepoints,
        group_ids=group_family_ids,
        group_labels=group_labels,
        dR2=dR2,
        trace_var=trace_var,
        beta_sig=beta_sig,
        beta_thr=beta_thr,
        family_p=family_p,
        real_max=real_max,          # per family: max|real beta| (the real max-statistic)
        fam_maxnull=fam_maxnull,    # per family x permutation: max|null beta| (the shared-shift null
                                    # distribution) -- pooled ACROSS channels by permutation index for
                                    # the array-level test in PlotArraySummary.py
        col_null=col_null,          # per column (lag) x permutation: SIGNED null beta (empty if
                                    # SAVE_COL_NULL=False) -- pooled across channels by permutation
                                    # index for the pooled mean-kernel test in AnalyzePooledKernels.py
        perm_shifts=perm_shifts,    # the shift schedule (identical across an array's channels)
        n_perm=N_PERM,
        alpha_sig=ALPHA,
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
