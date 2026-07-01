"""
=============================================================================================
Build the GLM design matrix for every session/channel.
=============================================================================================
Pipeline position: step 3 (see README.md). Consumes the per-channel regressors written by
GLM_input/Regressors.py (+ neural_data.npz from NeuralData.py) and produces, per channel:
    {designMatID}_dMatRaw_sparse.npz / _metadata.npz        (raw, all columns)
    {designMatID}_dMatProcessed_sparse.npz / _metadata.npz  (sparse columns dropped + normalised; fitted)
    {designMatID}_neural_downsampled.npz                    (LFP target at the design's row rate; FittingGLM input)

Runs over MANY session/channel pairs: by default it auto-discovers every
`<session>/channel<ch>_regressors` folder under `results_dir`; set SESSIONS / CHANNELS to pin a
subset. All the per-channel work lives in `process_channel()`, which is parallelised one worker
per channel across SLURM with acme (like Regressors.py / NeuralData.py).

Run in the warping env (needs acme: `conda install -c conda-forge esi-acme`).

Two parts (per channel):
  1. BUILD   - expand each regressor family into time-lagged kernels and concatenate -> raw matrix.
  2. FILTER  - drop near-empty columns and normalise -> processed matrix (what FittingGLM fits).
  + a diagnostic QR rank check that only *reports* redundant groups (not applied to the fit;
    the required redundancy sweep lives in RedundancySubsample.py / step 3a).

Regressor families:
  cognitive (per-trial boxcar, eventType=1): correct, wrong, diff_easy, diff_hard,
      movement_left, movement_right, state_<k> (one per cognitive state)
  events (impulse + kernel, eventType=3): trial_onset, stim_onset, reward_onset, block_onset,
      reaction_onset, saccade_onset, saccade_offset,
      {target, distractor, sky, mountain, grass}_in_RF_onset/offset
  dummy control (shuffled)
"""

import os
import sys
import re
import glob
import math
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, diags, save_npz
from scipy.linalg import qr as dense_qr

from acme import ParallelMap

sys.path.insert(1, '/mnt/cs/projects/MWzeronoise/Analysis/4Shivangi/code/functions/GLM_fitting')
from utils import makeLogical
from reg import makeDesignMatrix_noTrials

# ---------------------------
# Config
# ---------------------------
results_dir = '/cs/projects/MWzeronoise/Analysis/4Shivangi/Results/states_analysis/states_lfp/all_trials/full_length/GLM'

# Which session/channel pairs to build. None -> auto-discover from the results tree.
SESSIONS = None   # e.g. ['20230214']; None -> every session with channel*_regressors folders
CHANNELS = None   # e.g. [2]; applies to every session. None -> discover per session

# SLURM / acme parallelisation: one worker per channel (same config as Regressors.py / NeuralData.py)
SLURM_PARTITION = '16GBS'
MAX_WORKERS = 100
MEM_PER_WORKER = '16GB'

RANDOM_SEED = 7   # re-seeded per channel so the dummy shuffle is independent of loop order

PRE_TRIG_DUR = 0
POST_TRIG_DUR = 0

# Longest trial we model (LFP is zeroed after reward, so the trial->reward span bounds it)
maxTrialDur = 5000  # ms

# ---------------------------
# Reference-level drops (dummy-variable trap)
# ---------------------------
# Each per-trial categorical family below partitions EVERY trial, so its levels sum to the
# whole-trial boxcar -- which is exactly what trial_onset already is. Keeping trial_onset plus
# all levels of a family makes the design exactly rank-deficient (RedundancySubsample.py flags it),
# which breaks per-regressor interpretation and the RegressorContributions unique-variance analysis
# We keep trial_onset as the common baseline and drop ONE reference level per family. These are only 
# DROPPED FROM THE BUILD, not deleted -- to bring one back into the model, just remove it from this set 
# and rebuild.
DROP_REGRESSORS = {
    'diff_hard',       # difficulty family -> keep diff_easy as the modelled level
    'movement_right',  # movement family   -> keep movement_left
    'wrong',           # correctness family-> keep correct (deviation vs the wrong/error baseline)
    'state_1',         # state family: near-empty (2 trials) and part of the state partition
    'state_3',         # state family: near-empty (2 trials)
}

# ---------------------------
# Sample rate / downsampling  (YOU choose DOWNSAMPLE_FACTOR + the methods)
# ---------------------------
# The regressors + neural_data are saved at the LFP native rate (1 kHz). Kernel windows
# (pre_s/post_s) are turned into columns with `frameRate`, and the GLM lays one column per
# ROW of the matrix -- so frameRate MUST equal the matrix row rate or every window is off
# by the ratio. We therefore DERIVE frameRate = NATIVE_FS / DOWNSAMPLE_FACTOR below, so the
# windows stay correct in real seconds for whatever downsampling you pick.
NATIVE_FS = 1000           # Hz, rate the .npz regressors/neural_data were saved at
DOWNSAMPLE_FACTOR = 10     # 1 = keep 1 kHz (frameRate 1000, big matrix); 10 = 100 Hz (10x smaller)
EVENT_DOWNSAMPLE = 'any'   # bin event impulses: 'any' (keep every event) | 'sum' (count per bin)
NEURAL_DOWNSAMPLE = 'decimate' # bin the LFP: 'decimate' (anti-aliased) | 'mean' (bin-average) | 'subsample'


# ---------------------------
# Session/channel discovery
# ---------------------------
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


# --- downsampling helpers ---------------------------------------------------------------
# Event regressors are SPARSE 0/1 impulse trains. Never plain-subsample them: '[::D]' keeps
# only every Dth sample and silently DROPS any event that doesn't land on a kept sample.
# Instead bin into D-wide windows and OR-pool ('any') -> every event survives, only its
# timing is quantised to the D-sample (= 1 frame) bin. 'sum' keeps the event COUNT per bin.
# The LFP is a CONTINUOUS signal: 'mean' (bin-average, a crude low-pass) or 'decimate'
# (proper anti-alias filter then drop) avoid aliasing. 'subsample' ('[::D]') is offered but
# IS lossy for the LFP -- it folds frequencies above the new Nyquist (new_fs/2) back down.
def _truncate(v):
    n = (len(v) // DOWNSAMPLE_FACTOR) * DOWNSAMPLE_FACTOR   # make length a multiple of D
    return v[:n]


def downsample_event(v):
    if DOWNSAMPLE_FACTOR == 1:
        return v.astype(float)
    b = _truncate(v).reshape(-1, DOWNSAMPLE_FACTOR)
    if EVENT_DOWNSAMPLE == 'any':
        return (b.sum(axis=1) > 0).astype(float)   # 1 if the event fired anywhere in the bin
    if EVENT_DOWNSAMPLE == 'sum':
        return b.sum(axis=1).astype(float)          # number of events in the bin
    raise ValueError(f'unknown EVENT_DOWNSAMPLE {EVENT_DOWNSAMPLE!r}')


def downsample_neural(v):
    if DOWNSAMPLE_FACTOR == 1:
        return v
    if NEURAL_DOWNSAMPLE == 'mean':
        return _truncate(v).reshape(-1, DOWNSAMPLE_FACTOR).mean(axis=1)
    if NEURAL_DOWNSAMPLE == 'decimate':
        from scipy.signal import decimate
        return decimate(_truncate(v), DOWNSAMPLE_FACTOR, ftype='fir', zero_phase=True)
    if NEURAL_DOWNSAMPLE == 'subsample':
        return _truncate(v)[::DOWNSAMPLE_FACTOR]
    raise ValueError(f'unknown NEURAL_DOWNSAMPLE {NEURAL_DOWNSAMPLE!r}')


# =============================================================================================
# Per-channel pipeline
# =============================================================================================
def process_channel(channel, session):
    SESSION_ROOT = Path(results_dir) / session / f'channel{channel}_regressors'
    SAVE_PATH = SESSION_ROOT / 'results'
    SAVE_PATH.mkdir(parents=True, exist_ok=True)
    designMatID = f'{session}_channel{channel}'
    np.random.seed(RANDOM_SEED)   # deterministic dummy shuffle, independent of loop order
    print(f'\n========================= {designMatID} =========================')

    def load_reg(name):
        return np.load(os.path.join(SESSION_ROOT, f'{name}.npz'))['data']

    # =========================================================================================
    # Part 1 - BUILD the raw design matrix
    # =========================================================================================

    # --- load regressors + neural data ---
    neural_data = load_reg('neural_data').squeeze()

    trial_onset = load_reg('trial_onset')
    stim_onset = load_reg('stim_onset')
    reward_onset = load_reg('reward_onset')
    block_onset = load_reg('block_onset')
    reaction_onset = load_reg('reaction_onset')
    saccade_onset = load_reg('saccade_onset')
    saccade_offset = load_reg('saccade_offset')

    rf_cats = ['target', 'distractor', 'sky', 'mountain', 'grass']
    rf_onset = {c: load_reg(f'{c}_in_RF_onset') for c in rf_cats}
    rf_offset = {c: load_reg(f'{c}_in_RF_offset') for c in rf_cats}

    # --- downsample all sample-resolution arrays (LFP averaged, events OR-pooled) so the row
    # rate becomes NATIVE_FS/DOWNSAMPLE_FACTOR; frameRate is set to match below. Per-trial
    # cognitive/state regressors are NOT downsampled (they are per-trial, placed on trialTimes).
    neural_data = downsample_neural(neural_data)
    n_t = neural_data.shape[0]
    trial_onset = downsample_event(trial_onset)
    stim_onset = downsample_event(stim_onset)
    reward_onset = downsample_event(reward_onset)
    block_onset = downsample_event(block_onset)
    reaction_onset = downsample_event(reaction_onset)
    saccade_onset = downsample_event(saccade_onset)
    saccade_offset = downsample_event(saccade_offset)
    rf_onset = {c: downsample_event(rf_onset[c]) for c in rf_cats}
    rf_offset = {c: downsample_event(rf_offset[c]) for c in rf_cats}
    print(f'rate {NATIVE_FS} Hz / {DOWNSAMPLE_FACTOR} = {NATIVE_FS // DOWNSAMPLE_FACTOR} Hz -> '
          f'{n_t} rows (events: {EVENT_DOWNSAMPLE}, neural: {NEURAL_DOWNSAMPLE})')

    # per-trial binary (cognitive) regressors: the fixed set + however many state_<k> exist
    binary_labels = ['correct', 'wrong', 'diff_easy', 'diff_hard', 'movement_left', 'movement_right']
    state_labels = sorted(
        (os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(SESSION_ROOT, 'state_*.npz'))),
        key=lambda s: int(s.split('_')[1]))
    # drop the reference levels (see DROP_REGRESSORS) so the design stays full-rank
    allCogLabels = binary_labels + state_labels
    cogLabels = [lab for lab in allCogLabels if lab not in DROP_REGRESSORS]
    dropped = [lab for lab in allCogLabels if lab in DROP_REGRESSORS]
    if dropped:
        print(f'Dropping redundant reference regressors from the build: {dropped}')
    cog_data = {lab: load_reg(lab) for lab in cogLabels}

    opts = dict(
        skipIfExist=False, showOrthplot=False,
        preTrigDur=PRE_TRIG_DUR, postTrigDur=POST_TRIG_DUR,
        nrFolds=20, testFrac=0.1, removeAutoTrials=True, innateTask=False,
        preStimDur=0, postStimDur=0, preMoveDur=0, postMoveDur=0,
    )

    opts['postTrigDur'] = maxTrialDur / 1000  # seconds
    opts['frameRate'] = NATIVE_FS // DOWNSAMPLE_FACTOR  # MUST equal the matrix row rate (else windows are wrong)
    opts['trialDur'] = opts['preTrigDur'] + opts['postTrigDur']
    opts['framesPerTrial'] = math.ceil(opts['trialDur'] * opts['frameRate'] + 1)
    opts['preTrig'] = math.ceil(opts['frameRate'] * opts['preTrigDur'])
    opts['postTrig'] = math.ceil(opts['frameRate'] * opts['postTrigDur'])

    # Per-family pre-window in FRAMES (mPreTime). The kernel columns run lag = -mPreTime..+mPostTime,
    # so column c of a family = (c - pre_frames)/frameRate seconds from the event. Saved with the
    # design so AnalyzeRegressorContributions can put the kernel x-axis in seconds (event at 0).
    family_pre_frames = {}

    def set_kernel(pre_s, post_s):
        """Set the stim/move kernel window (seconds) used by makeDesignMatrix_noTrials."""
        opts['preStimDur'] = pre_s
        opts['postStimDur'] = post_s
        opts['stimPreTime'] = math.ceil(pre_s * opts['frameRate'])
        opts['stimPostTime'] = math.ceil(post_s * opts['frameRate'])
        opts['mPreTime'] = opts['stimPreTime']
        opts['mPostTime'] = opts['stimPostTime']

    def event_block(labels, arrays, pre_s, post_s):
        """Build an event (eventType=3) design block from sample-resolution event vectors."""
        set_kernel(pre_s, post_s)
        for lab in labels:
            family_pre_frames[lab] = opts['mPreTime']   # event kernel: lag -mPreTime..+mPostTime
        ev = np.zeros((n_t, len(labels)), dtype=bool)
        for i, a in enumerate(arrays):
            ev[:, i] = a
        return makeDesignMatrix_noTrials(ev, np.full(len(labels), 3, dtype=int), labels, opts)

    # --- 1a. cognitive regressors (per-trial boxcar over the whole trial) ---
    trialTimes = np.flatnonzero(trial_onset)
    eventType = np.ones(len(cogLabels), dtype=int)
    events = np.zeros((n_t, len(cogLabels)), dtype=bool)
    for i, lab in enumerate(cogLabels):
        events[:, i] = makeLogical(trialTimes[cog_data[lab].astype(bool)], n_t)

    print('Event counts per cognitive regressor:')
    for lab, col in zip(cogLabels, events.T):
        print(f'  {lab}: {int(col.sum())}')

    cogR, cogIdx = makeDesignMatrix_noTrials(events, eventType, cogLabels, opts)
    for lab in cogLabels:
        family_pre_frames[lab] = opts['preTrig']    # cognitive boxcar: lag -preTrig..+postTrig (preTrig=0)
    print(f'cogR shape: {cogR.shape}')

    # --- 1b. trial-event regressors ---
    # These are transient event responses, so use short peri-event kernels. The old whole-trial
    # windows (post = maxTrialDur) only had a real response in the first ~0.3 s; the multi-second
    # tail was noise that overfit in-sample and dragged down the cross-validated contribution
    # (RegressorContributions). Windows are short (~1 s) to capture the transient without the tail.
    trialOnsetLabels = ['trial_onset']
    trialOnsetR, trialOnsetIdx = event_block(trialOnsetLabels, [trial_onset],
                                             pre_s=0, post_s=1.0)

    stimOnsetLabels = ['stim_onset']
    stimOnsetR, stimOnsetIdx = event_block(stimOnsetLabels, [stim_onset],
                                           pre_s=0.2, post_s=1.0)

    rewardOnsetLabels = ['reward_onset']
    rewardOnsetR, rewardOnsetIdx = event_block(rewardOnsetLabels, [reward_onset],
                                               pre_s=0.5, post_s=0)

    blockOnsetLabels = ['block_onset']
    blockOnsetR, blockOnsetIdx = event_block(blockOnsetLabels, [block_onset],
                                             pre_s=(maxTrialDur / 1000), post_s=0)

    # reaction time: response-locked kernel (tune the window as needed)
    reactionLabels = ['reaction_onset']
    reactionR, reactionIdx = event_block(reactionLabels, [reaction_onset],
                                         pre_s=0.3, post_s=0.5)

    # saccades (microsaccades already excluded upstream): short eye-movement-locked kernels
    saccadeLabels = ['saccade_onset', 'saccade_offset']
    saccadeR, saccadeIdx = event_block(saccadeLabels, [saccade_onset, saccade_offset],
                                       pre_s=0.2, post_s=0.3)

    # --- 1c. RF regressors ---
    # Leaves (target / distractor) are discrete stimuli entering/leaving the RF -> transient
    # response, so they keep a time-lagged onset/offset kernel. The background bands
    # (sky / mountain / grass) are a static skybox the gaze dwells on; they get a 0-width
    # window (single lag-0 impulse, pre=post=0) instead of a kernel.
    leaf_cats = ['target', 'distractor']
    bg_cats = ['sky', 'mountain', 'grass']

    # leaves: onset/offset with kernels
    RFLeafOnsetLabels = [f'{c}_in_RF_onset' for c in leaf_cats]
    RFLeafOnsetR, RFLeafOnsetIdx = event_block(RFLeafOnsetLabels, [rf_onset[c] for c in leaf_cats],
                                               pre_s=0.1, post_s=0.5)
    RFLeafOffsetLabels = [f'{c}_in_RF_offset' for c in leaf_cats]
    RFLeafOffsetR, RFLeafOffsetIdx = event_block(RFLeafOffsetLabels, [rf_offset[c] for c in leaf_cats],
                                                 pre_s=0.1, post_s=0)

    # background bands: 0-width window (lag-0 impulse only, no kernel)
    RFBgOnsetLabels = [f'{c}_in_RF_onset' for c in bg_cats]
    RFBgOnsetR, RFBgOnsetIdx = event_block(RFBgOnsetLabels, [rf_onset[c] for c in bg_cats],
                                           pre_s=0, post_s=0)
    RFBgOffsetLabels = [f'{c}_in_RF_offset' for c in bg_cats]
    RFBgOffsetR, RFBgOffsetIdx = event_block(RFBgOffsetLabels, [rf_offset[c] for c in bg_cats],
                                             pre_s=0, post_s=0)

    # --- 1d. dummy control regressor (shuffled in time) ---
    dummyLabel = ['dummy']
    dummyEvents = makeLogical(trialTimes, n_t)
    dummyR, dummyIdx = makeDesignMatrix_noTrials(dummyEvents[:, None], [1], dummyLabel, opts)
    family_pre_frames['dummy'] = opts['preTrig']    # same eventType=1 kernel as the cognitive boxcars
    for iCol in range(dummyR.shape[1]):
        dummyR[:, iCol] = dummyR[np.random.permutation(dummyR.shape[0]), iCol]

    # --- 1e. concatenate into the full (raw) design matrix ---
    blocks = [
        (cogR, cogIdx, cogLabels),
        (trialOnsetR, trialOnsetIdx, trialOnsetLabels),
        (stimOnsetR, stimOnsetIdx, stimOnsetLabels),
        (rewardOnsetR, rewardOnsetIdx, rewardOnsetLabels),
        (blockOnsetR, blockOnsetIdx, blockOnsetLabels),
        (reactionR, reactionIdx, reactionLabels),
        (saccadeR, saccadeIdx, saccadeLabels),
        (RFLeafOnsetR, RFLeafOnsetIdx, RFLeafOnsetLabels),
        (RFLeafOffsetR, RFLeafOffsetIdx, RFLeafOffsetLabels),
        (RFBgOnsetR, RFBgOnsetIdx, RFBgOnsetLabels),
        (RFBgOffsetR, RFBgOffsetIdx, RFBgOffsetLabels),
        (dummyR, dummyIdx, dummyLabel),
    ]

    fullR = np.concatenate([b[0] for b in blocks], axis=1)

    regIdx_parts, regLabels, running_max = [], [], 0
    for _, idx, labels in blocks:
        regIdx_parts.append(idx + running_max)
        running_max = int(np.max(idx + running_max))
        regLabels += labels
    regIdx = np.concatenate(regIdx_parts, axis=0)

    print(f'Full (raw) design matrix shape: {fullR.shape}, {len(regLabels)} groups')
    print('Regressor groups:', regLabels)

    fullR_sparse = csr_matrix(fullR)
    save_npz(os.path.join(SAVE_PATH, f'{designMatID}_dMatRaw_sparse.npz'), fullR_sparse)
    np.savez(os.path.join(SAVE_PATH, f'{designMatID}_dMatRaw_metadata.npz'),
             regIdx=regIdx, regLabels=np.array(regLabels, dtype=object))
    print(f'Saved raw -> {SAVE_PATH}/{designMatID}_dMatRaw_sparse.npz')

    # =========================================================================================
    # Part 2 - FILTER: drop near-empty columns + normalise -> processed matrix (this is what we fit)
    # =========================================================================================
    # Drop near-empty columns (too few events at that lag). Threshold to tune (10 rejects nothing).
    col_sum = np.array(np.abs(fullR_sparse).sum(axis=0)).ravel()
    rejIdx = col_sum < 30 #100
    Rkeep = fullR_sparse[:, ~rejIdx]

    col_norm = np.sqrt(Rkeep.power(2).sum(axis=0)).A1
    col_norm[col_norm == 0] = 1.0
    fullR_proc = Rkeep @ diags(1 / col_norm)
    regIdx_kept = regIdx[~rejIdx]

    print(f'Processed matrix shape: {fullR_proc.shape}, rejected {np.sum(rejIdx)}/{len(rejIdx)} sparse columns')

    # per-group breakdown of which regressors lost columns to the sparse-column filter
    print('Columns dropped by the sparse-column filter (col_sum < 100):')
    any_dropped = False
    for gid in np.unique(regIdx):
        grp = regIdx == gid
        n_drop = int(np.sum(rejIdx & grp))
        if n_drop:
            any_dropped = True
            n_tot = int(np.sum(grp))
            note = '  <-- group fully removed' if n_drop == n_tot else ''
            print(f'  {regLabels[int(gid) - 1]}: dropped {n_drop}/{n_tot} columns{note}')
    if not any_dropped:
        print('  none')

    regLabelsNew = [regLabels[int(r) - 1] for r in np.unique(regIdx_kept)]

    # per-family pre-window (frames) aligned to regLabelsNew, + frame rate, so kernel plots can put
    # their x-axis in seconds relative to the event (lag column c -> (c - pre_frames)/frame_rate s).
    lag_pre_frames = np.array([family_pre_frames.get(lab, 0) for lab in regLabelsNew], dtype=int)

    save_npz(os.path.join(SAVE_PATH, f'{designMatID}_dMatProcessed_sparse.npz'), fullR_proc)
    np.savez(os.path.join(SAVE_PATH, f'{designMatID}_dMatProcessed_metadata.npz'),
             regIdx=regIdx_kept, regLabels=np.array(regLabelsNew, dtype=object),
             lag_pre_frames=lag_pre_frames, frame_rate=NATIVE_FS // DOWNSAMPLE_FACTOR)
    print(f'Saved processed -> {SAVE_PATH}/{designMatID}_dMatProcessed_sparse.npz')

    # Save the DOWNSAMPLED neural target so FittingGLM fits against a target aligned to the design's
    # row rate
    np.savez(os.path.join(SAVE_PATH, f'{designMatID}_neural_downsampled.npz'),
             data=neural_data, frame_rate=NATIVE_FS // DOWNSAMPLE_FACTOR)
    print(f'Saved downsampled neural target ({neural_data.shape[0]} rows) -> '
          f'{SAVE_PATH}/{designMatID}_neural_downsampled.npz')

    # =========================================================================================
    # Diagnostic - QR rank check via the Gram matrix (reports redundant groups; not fitted)
    # =========================================================================================
    # Rank deficiency = a column is an exact linear combination of others (coeffs not unique);
    # multicollinearity = columns highly (not exactly) correlated (coeffs unstable). QR detects the
    # former; ridge handles the latter. The full sweep over downscale factors is RedundancySubsample.py.
    #
    # We check the FITTED (processed) matrix via the Gram matrix G = X^T X: a sparse matmul giving a
    # small (cols x cols) dense matrix, so the QR is tiny regardless of how many rows we feed it.
    # rank(G) = rank(X), and a column that is a linear combination of others in X is the SAME
    # combination of columns in G, so a column-pivoted QR of G reveals exactly the redundant columns.
    # (Forming G squares the condition number, so this targets EXACT dependence; near-dependence /
    # multicollinearity is left to ridge.)
    #
    # Row-thinning: only worth it when the matrix was built at the native 1 kHz rate (no upstream
    # decimation -> millions of rows make forming G slow). If DOWNSAMPLE_FACTOR already shrank it,
    # keep every row. We clamp the factor so kept rows stay >> columns -- row-subsampling can only
    # LOWER rank, so thinning below ~n_cols rows would manufacture FALSE redundancy on its own.
    n_rows, n_cols = fullR_proc.shape
    qr_factor = 100 if DOWNSAMPLE_FACTOR == 1 else 1      # thin only if not decimated upstream
    qr_factor = min(qr_factor, max(1, n_rows // (2 * n_cols)))  # safety: keep rows >> columns
    rows = np.arange(0, n_rows, qr_factor)
    Xqr = fullR_proc[rows]                                # full rows when already downsampled
    G = (Xqr.T @ Xqr).toarray()                           # (n_cols x n_cols) Gram
    _, Rqr, piv = dense_qr(G, mode='economic', pivoting=True)

    diagR = np.abs(np.diag(Rqr))
    threshold = max(Xqr.shape) * np.spacing(diagR[0]) if diagR.size else 0.0
    redundant = diagR <= threshold
    print(f'Rank check on {len(rows)}/{n_rows} rows (qr_factor={qr_factor}), {n_cols} columns')
    if np.any(redundant):
        redundant_cols = piv[redundant]                   # column indices into fullR_proc
        print(f'Design matrix is rank-deficient: {int(redundant.sum())}/{n_cols} redundant columns.')
        zero_regs = np.unique(regIdx_kept[redundant_cols])
        print('Regressor groups contributing redundant columns:',
              [regLabels[int(r) - 1] for r in zero_regs])
    else:
        print('No fully redundant regressor groups detected (full-matrix Gram QR).')


# =============================================================================================
# Driver: parallelise channels across the cluster (one worker per channel, per session)
# =============================================================================================
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

        with ParallelMap(process_channel, channels, session,
                         n_inputs=len(channels),
                         partition=SLURM_PARTITION,
                         n_workers=n_workers,
                         mem_per_worker=MEM_PER_WORKER,
                         setup_timeout=600,   # busy cluster: wait up to 10 min for SLURM to allocate
                         write_worker_results=False,   # workers save their own npz; nothing to collect
                         setup_interactive=False) as pmap:
            pmap.compute()
