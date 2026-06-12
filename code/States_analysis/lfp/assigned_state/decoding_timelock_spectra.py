"""
Summary
-------
This script answers a simple yes/no question: does the time-domain LFP (ERP)
carry information about the cognitive state of the animal?

This script asks the direct question -- "can we decode the state from the neural signal
at all, and how well?" -- by training a cross-validated classifier on the whole trial and
testing its accuracy against a label-shuffle null.

We have one discrete state label per trial, so this is a standard decoding
problem:
- X = a per-trial feature vector. Two representations are supported (configurable
      via FEATURE_SETS):
        * 'erp'       -- the full per-trial ERP (LFP amplitude across the
                         [-0.2, 0.9]s window), shape (nTrials, nTimePoints). The
                         whole trial is used jointly, so temporal shape (peaks,
                         slopes, latencies) contributes -- not just isolated samples.
        * 'spectrum'  -- full per-trial power spectrum (2-100 Hz) via syncopy
                         mtmfft (same estimator as erp_spectra_stats.py), shape
                         (nTrials, nFreq), keeping full frequency resolution.
- y = the state label per trial. Which states are decoded is configurable
      (STATES_TO_DECODE), so you can target specific or most-separable states.

Classifier: shrinkage LDA (solver='lsqr', shrinkage='auto') inside a pipeline
with feature standardisation. Shrinkage keeps the covariance estimate stable
when there are many time points relative to trials. It is a linear model.

Evaluation: stratified k-fold cross-validation with balanced accuracy (states
are typically imbalanced, so balanced accuracy compares fairly against a chance
level of 1 / nStates). Alongside accuracy we report the mutual information
I(state ; predicted-state) in bits, computed from the cross-validated confusion
matrix. By the data-processing inequality this is a clean *lower bound* on
I(state ; LFP) -- "the LFP carries at least this many bits about state" -- and we
express it as a fraction of the total state entropy H(state). Significance for
both metrics comes from a permutation test that shuffles the state labels and
re-runs the full CV; the same shuffle null also de-biases the MI estimate
(confusion-matrix MI is positively biased at finite sample size), so we report
the bias-corrected MI = observed - null-mean.

Analyses are run at three levels (mirroring the companion scripts):
- single channel,
- array (channels grouped into 6 arrays, mean ERP across channels per trial),
- combined arrays (arrays 1 to 3 merged, others separate).

Outputs:
- `decoding_summary.csv`: one row per feature/channel/array with balanced
  accuracy, chance level, permutation p-value, bias-corrected MI (bits and as a
  fraction of state entropy), MI p-value, and trial/state counts.
- Per-array bar plots of decoding accuracy vs chance with significance markers.
- Per-array bar plots of state information (% of state entropy) with markers.
- Time/frequency-resolved decoding curves (array level) localising which time
  points (erp) and frequencies (spectrum) carry state information, with
  max-statistic permutation correction across bins.
"""

# -----------------------------
# Imports
# -----------------------------
import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import syncopy as spy
import json
from scipy.signal import decimate
from joblib import Parallel, delayed
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

# balanced_accuracy warns when a CV fold's test set lacks a class the decoder
# still predicts -- expected with imbalanced multi-class group CV and harmless.
warnings.filterwarnings(
    "ignore", message="y_pred contains classes not in y_true")

# custom path for parse_logfile (kept for parity with companion scripts)
sys.path.insert(1, '/mnt/cs/projects/MWzeronoise/Analysis/4Shivangi/code/functions/unreal_logfile')
from parse_logfile import TextLog  # noqa: E402,F401

# -----------------------------
# User Config
# -----------------------------
lfp_data_dir = '/cs/projects/MWzeronoise/Analysis/4Shivangi/Datasets/neural_data/stimAalign_cut/clean_full_length'
trial_info_dir = '/cs/projects/MWzeronoise/Analysis/4Shivangi/Datasets/neural_data/stimAalign_cut/full_length'
states_data_dir = '/cs/projects/MWzeronoise/Analysis/4Shivangi/Datasets/states_analysis'

LATENCY = [-0.2, 0.9]
WIN_TAG = f"{int(round(abs(LATENCY[0]) * 1000))}_{int(round(abs(LATENCY[1]) * 1000))}"
# Output roots. The state-tagged leaf (e.g. ".../timelock_spectra/states_0_2") is
# appended below once STATES_TO_DECODE is known.
output_root = f'/cs/projects/MWzeronoise/Analysis/4Shivangi/plots/states_lfp/all_trials/{WIN_TAG}/decoding/timelock_spectra'
results_dir = '/mnt/cs/projects/MWzeronoise/Analysis/4Shivangi/Results/states_analysis/states_lfp'
results_root = os.path.join(results_dir, "all_trials", WIN_TAG, "decoding", "timelock_spectra")

sessions = ['20230203', '20230208', '20230209', '20230213', '20230214']
N_STATES_TO_USE = 4       # used only when STATES_TO_DECODE is None: take the
                          # first N states (by sorted label) available in a session
STATES_TO_DECODE = [0, 2] # explicit states to decode, e.g. [0, 1, 2] or [0, 2].
                          # None -> fall back to the first N_STATES_TO_USE states.

# State-tagged leaf folder: explicit states -> "states_0_2"; None -> "states_first4".
STATE_TAG = ('_'.join(str(s) for s in STATES_TO_DECODE)
             if STATES_TO_DECODE is not None else f'first{N_STATES_TO_USE}')
output_dir = os.path.join(output_root, f'states_{STATE_TAG}')
results_data_dir = os.path.join(results_root, f'states_{STATE_TAG}')
os.makedirs(output_dir, exist_ok=True)
os.makedirs(results_data_dir, exist_ok=True)
FEATURE_SETS = ['erp', 'spectrum']  # representations to decode: 'erp' (evoked
                          # waveform, time-domain) and/or 'spectrum'. Each is
                          # decoded independently.
# Spectral analysis (syncopy mtmfft), mirroring erp_spectra_stats.py: a
# Hann-tapered low band and a multitaper high band, concatenated into one
# per-trial power spectrum.
SPEC_LOW_FOILIM = [2, 30]
SPEC_HIGH_FOILIM = [30, 100]
SPEC_HIGH_TAPSMOFRQ = 4
# Time/frequency-resolved decoding: decode state from each bin separately to
# localise WHICH time points ('erp') / frequencies ('spectrum') carry state
# information. Array level only (array-mean per bin), keeping the same session-
# grouped CV; multiple comparisons across bins are controlled with a
# max-statistic permutation null (cf. erp_spectra_stats.py).
RESOLVED_FEATURES = ['erp', 'spectrum']  # features to localise; [] to skip
RESOLVED_ALPHA = 0.05
n_perms = 200            # label shuffles for the decoding null / p-value
n_splits = 5             # stratified CV folds
N_JOBS = -1              # parallel workers across channels/arrays (-1 = all cores)
ORIG_FS = 1000           # native LFP sampling rate (Hz)
TARGET_FS = 200          # decode at this rate; ERP content is < ~40 Hz, so 1000 Hz
                         # is hugely oversampled (redundant features, ill-conditioned
                         # LDA covariance).
DECIM_Q = ORIG_FS // TARGET_FS  # decimation factor (5)
PATTERN_CHANNEL = None    # channel to plot the LDA decoding pattern for; if None,
                          # the most significant per-channel result is used.
min_trials_per_state = 5  # require at least this many trials/state to keep a state
RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)


# load states info for all sessions
state_probs = np.load(f'{states_data_dir}/foraging_shivangi_no_sess1_clipped_state_assignments.npy')
with open(f'{states_data_dir}/foraging_shivangi_no_sess1_clipped_session_index.json') as f:
    session_index = json.load(f)

session_to_probs = {}
for sess in session_index:
    session_id = sess['session_id']
    session_date = session_id.split('_')[1]
    session_to_probs[session_date] = state_probs[
        sess['start_idx']: sess['end_idx'] + 1
    ]

# -----------------------------
# Helper functions
# -----------------------------
def remove_nan_trials_channels(datas):
    trial_mask = [not np.all(np.isnan(tr)) for tr in datas.trials]
    if not any(trial_mask):
        return None, []
    cfg = spy.StructDict(trials=np.where(trial_mask)[0])
    datas_clean = spy.selectdata(cfg, datas)
    trial_stack = np.stack(datas_clean.trials, axis=0)
    valid_ch_idx = np.where(~np.all(np.isnan(trial_stack), axis=(0, 1)))[0]
    if len(valid_ch_idx) == 0:
        return None, []
    cfg = spy.StructDict(channel=valid_ch_idx)
    datas_clean = spy.selectdata(cfg, datas_clean)
    valid_channels = [datas.channel[i] for i in valid_ch_idx]
    return datas_clean, valid_channels

def ensure_trialindex_in_trialdefinition(datalfp):
    if datalfp.trialdefinition.shape[1] < 4:
        nTrials = datalfp.trialdefinition.shape[0]
        datalfp.trialdefinition = np.hstack(
            (datalfp.trialdefinition, np.arange(nTrials).reshape(-1, 1))
        )

def extract_power_trials(freq_analysis):
    """Pull per-trial power out of a syncopy freqanalysis result as
    (nTrials, nFreq, nCh). Mirrors erp_spectra_stats.py so the two scripts use
    identical spectral handling. Returns (None, None, None) on an unexpected shape."""
    try:
        freqs = getattr(freq_analysis, 'freq', None)
        channels = getattr(freq_analysis, 'channel', None)
        candidate = np.squeeze(freq_analysis.trials)
        if candidate.ndim == 3:
            return candidate, freqs, channels
        if candidate.ndim == 2 and freqs is not None and channels is not None:
            if candidate.shape[1] == len(freqs) * len(channels):
                reshaped = candidate.reshape(
                    (candidate.shape[0], len(freqs), len(channels)))
                return reshaped, freqs, channels
        return None, None, None
    except Exception as e:
        print("extract_power_trials error:", e)
        return None, None, None

def compute_spectrum(datas_state_clean):
    """Per-trial power spectrum via syncopy mtmfft, matching erp_spectra_stats.py:
    a Hann-tapered low band and a multitaper high band, concatenated.
    Returns (power_trials (nTrials, nFreq, nCh), freqs) or (None, None)."""
    cfg_low = spy.StructDict(method='mtmfft', foilim=SPEC_LOW_FOILIM, out='pow',
                             keeptrials=True, taper='hann')
    low_power, freqs_low, _ = extract_power_trials(
        spy.freqanalysis(datas_state_clean, cfg_low))
    cfg_high = spy.StructDict(method='mtmfft', foilim=SPEC_HIGH_FOILIM, out='pow',
                              keeptrials=True, tapsmofrq=SPEC_HIGH_TAPSMOFRQ)
    high_power, freqs_high, _ = extract_power_trials(
        spy.freqanalysis(datas_state_clean, cfg_high))
    if low_power is None or high_power is None:
        return None, None
    power_trials = np.concatenate((low_power, high_power), axis=1)
    freqs = np.concatenate((freqs_low, freqs_high))
    # log10 stabilises the heavy-tailed power distribution for the linear LDA.
    return np.log10(power_trials + 1e-20), freqs

def state_entropy_bits(y):
    """Shannon entropy H(state) in bits from the empirical label distribution."""
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log2(p)))

def confusion_mi_bits(y_true, y_pred, labels):
    """Mutual information I(y_true ; y_pred) in bits from the joint confusion
    matrix. This is the information transmitted through the decoder and, by the
    data-processing inequality, a lower bound on I(state ; features)."""
    cm = confusion_matrix(y_true, y_pred, labels=labels).astype(float)
    total = cm.sum()
    if total == 0:
        return 0.0
    p_joint = cm / total
    p_true = p_joint.sum(axis=1, keepdims=True)   # P(y_true)
    p_pred = p_joint.sum(axis=0, keepdims=True)   # P(y_pred)
    p_indep = p_true @ p_pred
    nz = (p_joint > 0) & (p_indep > 0)
    return float(np.sum(p_joint[nz] * np.log2(p_joint[nz] / p_indep[nz])))

def decode_state(X, y, groups, n_perms=200, n_splits=5, random_state=42):
    """
    Cross-validated decoding of discrete labels y from features X, with a
    label-shuffle permutation test.

    Uses StratifiedGroupKFold with `groups` = session id, so every session is
    held entirely out in its test fold. The decoder therefore has to generalise
    ACROSS sessions: above-chance accuracy means the ERP tracks state, not just
    session identity (which pooled CV could exploit).

    Returns dict with balanced accuracy, chance level, permutation p-value,
    and the null distribution, or None if there are too few trials/states/groups.
    """
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return None
    n_groups = len(np.unique(groups))
    # group CV needs >=2 sessions, and at most one group can be held out per fold
    if n_groups < 2:
        return None
    # need enough samples in the smallest class to populate folds
    folds = int(min(n_splits, n_groups, counts.min()))
    if folds < 2:
        return None

    clf = make_pipeline(
        StandardScaler(),
        LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    )

    def cv_metrics(y_use):
        # single-threaded here: parallelism happens across channels/arrays in the
        # outer loop, so nested n_jobs would oversubscribe the node. A fresh CV
        # object per call keeps splits reproducible.
        cv = StratifiedGroupKFold(n_splits=folds, shuffle=True,
                                  random_state=random_state)
        y_pred = cross_val_predict(clf, X, y_use, groups=groups, cv=cv, n_jobs=1)
        acc = balanced_accuracy_score(y_use, y_pred)
        mi = confusion_mi_bits(y_use, y_pred, classes)
        return acc, mi

    # Observed metrics from the real labels.
    acc_obs, mi_obs = cv_metrics(y)

    # Label-shuffle null: re-run the full CV on permuted labels. The same null
    # serves both the accuracy p-value and the MI bias correction (confusion-
    # matrix MI is positively biased at finite N, so the null mean estimates that
    # bias).
    perm_rng = np.random.default_rng(random_state)
    perm_acc = np.empty(n_perms)
    perm_mi = np.empty(n_perms)
    for i in range(n_perms):
        perm_acc[i], perm_mi[i] = cv_metrics(perm_rng.permutation(y))

    p_acc = (1 + np.sum(perm_acc >= acc_obs)) / (n_perms + 1)
    p_mi = (1 + np.sum(perm_mi >= mi_obs)) / (n_perms + 1)
    mi_null_mean = float(np.mean(perm_mi))
    mi_corrected = max(0.0, mi_obs - mi_null_mean)
    H_state = state_entropy_bits(y)

    return {
        'balanced_accuracy': float(acc_obs),
        'chance': float(1.0 / len(classes)),
        'p_value': float(p_acc),
        'perm_mean': float(np.mean(perm_acc)),
        'perm_scores': perm_acc,
        'mi_bits_raw': float(mi_obs),
        'mi_bits': float(mi_corrected),
        'mi_null_mean': mi_null_mean,
        'state_entropy_bits': float(H_state),
        'mi_frac_entropy': float(mi_corrected / H_state) if H_state > 0 else 0.0,
        'mi_p_value': float(p_mi),
        'n_folds': folds,
        'n_groups': int(n_groups),
    }

# -----------------------------
# Main data collection
# -----------------------------
# Per state, a list (one entry per session) of
# {'trials': (nTrials, nTimePoints, nChannels), 'x': time_vec, 'channels': [...]}.
state_data_timelock = {}

for session_name in sessions:
    print(f"\n=== Processing session {session_name} ===")
    lfp_path = os.path.join(lfp_data_dir, session_name, 'Cleaned_lfp_FT.spy')
    trial_info_path = os.path.join(trial_info_dir, session_name, 'Trial_Info.pkl')
    if not os.path.exists(lfp_path) or not os.path.exists(trial_info_path):
        continue

    # state info
    predicted_states = session_to_probs[session_name]
    trial_info_df = pd.read_pickle(trial_info_path)
    trial_info_df.iloc[:, 0] = (trial_info_df.iloc[:, 0] - 1000).astype('Int64')
    stim_df = pd.DataFrame({'TrialIndex': np.arange(len(predicted_states)), 'States': predicted_states})
    combined_df = pd.merge(trial_info_df, stim_df, left_on='Trial_Number', right_on='TrialIndex', how='inner')

    # load LFP data
    datalfp = spy.load(lfp_path)
    ensure_trialindex_in_trialdefinition(datalfp)
    cfg = spy.StructDict(latency=LATENCY)
    data = spy.selectdata(cfg, datalfp)
    selected_trials = data.trialdefinition[:, 3].astype(int)
    states_trial_info_filt = combined_df[combined_df['TrialIndex'].isin(selected_trials)]
    available_states = np.sort(np.unique(states_trial_info_filt['States'].to_numpy()))
    if STATES_TO_DECODE is not None:
        unique_states = np.array([s for s in available_states if s in STATES_TO_DECODE])
    else:
        unique_states = available_states[:N_STATES_TO_USE]

    for state_value in unique_states:
        cfg_sel = spy.StructDict(trials=np.where(states_trial_info_filt['States'] == state_value)[0])
        datas_state = spy.selectdata(cfg_sel, data)
        datas_state_clean, valid_channels = remove_nan_trials_channels(datas_state)
        if datas_state_clean is None:
            continue

        trials_array = np.stack(datas_state_clean.trials, axis=0)  # (nTrials, nTime, nCh)
        time_vec = datas_state_clean.time[0]

        # Full per-trial power spectrum (syncopy mtmfft) -> (nTrials, nFreq, nCh).
        spectrum_array, freqs = compute_spectrum(datas_state_clean)
        if spectrum_array is None:
            continue

        # Downsample the ERP along the time axis (data is already low-pass
        # filtered; decimate's anti-alias filter is extra safety). Slicing the
        # time vector with the same stride keeps it aligned with the decimated
        # samples.
        if DECIM_Q > 1:
            erp_array = decimate(trials_array, DECIM_Q, axis=1, ftype='fir')
            time_vec = time_vec[::DECIM_Q][:erp_array.shape[1]]
        else:
            erp_array = trials_array
        if erp_array.shape[0] < min_trials_per_state:
            continue

        if state_value not in state_data_timelock:
            state_data_timelock[state_value] = []
        state_data_timelock[state_value].append(
            {'erp': erp_array, 'spectrum': spectrum_array, 'x': time_vec,
             'freqs': freqs, 'channels': valid_channels, 'session': session_name})


# -----------------------------
# Pooling helpers (combine all sessions)
# -----------------------------          
def pool_channel(store, ch_name, feature):
    """Pool single-channel trials for one feature ('erp' or 'spectrum') across
    states/sessions. Feature arrays are (nTrials, nFeat, nCh), so the per-trial
    vector is (nTrials, nFeat) -- time points for 'erp', frequencies for 'spectrum'.
    Returns X (nTrials, nFeat), y (state per trial), groups (session per trial)."""
    X_parts, y_parts, g_parts = [], [], []
    for state_value in sorted(store.keys()):
        for sess in store[state_value]:
            if ch_name not in sess['channels']:
                continue
            ch_idx = sess['channels'].index(ch_name)
            n = sess[feature].shape[0]
            X_parts.append(sess[feature][:, :, ch_idx])  # (nTrials, nFeat)
            y_parts.append(np.full(n, state_value))
            g_parts.append(np.full(n, sess['session']))
    if not X_parts:
        return None, None, None
    return (np.concatenate(X_parts, axis=0), np.concatenate(y_parts),
            np.concatenate(g_parts))

def pool_array(store, ch_group, feature):
    """Pool array-level trials for one feature: per-trial vector is the mean
    feature across the array's channels (matches erp_spectra_stats.py array
    level). 'erp' -> mean ERP over time; 'spectrum' -> mean log power per frequency.
    Returns X (nTrials, nFeat), y (state per trial), groups (session per trial)."""
    X_parts, y_parts, g_parts = [], [], []
    for state_value in sorted(store.keys()):
        for sess in store[state_value]:
            ch_valid = [c for c in ch_group if c in sess['channels']]
            if not ch_valid:
                continue
            ch_idx = [sess['channels'].index(c) for c in ch_valid]
            n = sess[feature].shape[0]
            X_parts.append(np.mean(sess[feature][:, :, ch_idx], axis=2))  # (nTrials, nFeat)
            y_parts.append(np.full(n, state_value))
            g_parts.append(np.full(n, sess['session']))
    if not X_parts:
        return None, None, None
    return (np.concatenate(X_parts, axis=0), np.concatenate(y_parts),
            np.concatenate(g_parts))

def clean_xy(X, y, groups):
    """Drop trials with any non-finite feature (LDA cannot handle NaNs)."""
    good = ~np.any(~np.isfinite(X), axis=1)
    return X[good], y[good], groups[good]

def bh_fdr(pvals):
    """Benjamini-Hochberg FDR-adjusted p-values (q-values).

    Controls the expected *proportion* of false positives among the hits, which
    is the right error notion for a many-channel screen (vs. Bonferroni, which
    controls any single false positive and is far too conservative here). Valid
    under independence or positive dependence -- the regime of spatially
    correlated channels. Implemented directly to avoid a statsmodels dependency.
    """
    p = np.asarray(pvals, dtype=float)
    m = p.size
    order = np.argsort(p)
    ranks = np.arange(1, m + 1)
    q = p[order] * m / ranks
    # enforce monotonicity from the largest p downward, then clip to 1
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(m, dtype=float)
    out[order] = np.clip(q, 0, 1)
    return out

def decode_job(X, y, groups, level, array_index, name, feature):
    """Clean, decode, and return a summary row (or None). Pure function so it can
    run in a worker process under joblib.Parallel."""
    # joblib workers are fresh processes that don't re-run the module-level
    # filter, so re-apply it here so the warning is silenced inside each worker.
    warnings.filterwarnings(
        "ignore", message="y_pred contains classes not in y_true")
    if X is None:
        return None
    X, y, groups = clean_xy(X, y, groups)
    if X.shape[0] < min_trials_per_state * 2 or len(np.unique(y)) < 2:
        return None
    res = decode_state(X, y, groups, n_perms=n_perms, n_splits=n_splits,
                       random_state=RANDOM_STATE)
    if res is None:
        return None
    return {
        'feature': feature, 'level': level, 'array_index': array_index,
        'name': name,
        'n_trials': int(X.shape[0]), 'n_states': int(len(np.unique(y))),
        'n_sessions': res['n_groups'],
        'balanced_accuracy': res['balanced_accuracy'], 'chance': res['chance'],
        'perm_mean': res['perm_mean'], 'p_value': res['p_value'],
        'mi_bits': res['mi_bits'], 'mi_bits_raw': res['mi_bits_raw'],
        'mi_null_mean': res['mi_null_mean'],
        'state_entropy_bits': res['state_entropy_bits'],
        'mi_frac_entropy': res['mi_frac_entropy'], 'mi_p_value': res['mi_p_value'],
        'n_folds': res['n_folds'],
    }

def lda_activation_patterns(X, y):
    """Fit the decoding pipeline on all trials and return interpretable
    activation patterns over time -- one per state -- showing *which* part of
    the ERP the decoder uses to read out each state.

    The raw LDA weights are a backward (discriminative) model and must NOT be
    read directly as 'where the signal is': a large weight can sit on a feature
    purely to cancel noise. Following Haufe et al. (2014), we map the weights W
    to activation patterns A = Cov(X_std) @ W, which *are* interpretable as the
    signal the decoder responds to. Patterns are rescaled back to the original
    LFP units so the y-axis is meaningful.
    """
    scaler = StandardScaler().fit(X)
    X_std = scaler.transform(X)
    lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(X_std, y)
    W = np.atleast_2d(lda.coef_)               # (nClasses, nTime), standardised space
    cov_std = np.cov(X_std, rowvar=False)      # (nTime, nTime)
    A_std = cov_std @ W.T                       # (nTime, nClasses), Haufe pattern
    A = A_std * scaler.scale_[:, None]          # undo standardisation -> original units
    return lda.classes_, A.T                    # classes, patterns (nClasses, nTime)

def resolved_decode(X, y, groups, n_perms, n_splits, random_state, alpha):
    """Per-bin (time- or frequency-resolved) decoding to localise WHICH bins
    carry state information.

    X: (nTrials, nBins) -- the array-mean signal, one column per time point
    ('erp') or frequency ('spectrum'). Each bin is decoded independently with the
    same session-grouped CV as the joint decoder, giving per-bin balanced accuracy
    and confusion-matrix MI. Multiple comparisons across bins are controlled with a
    max-statistic permutation null (cf. erp_spectra_stats.py): each label shuffle
    recomputes the whole per-bin curve and contributes its maximum, so a bin is
    significant if it exceeds the (1-alpha) quantile of that max distribution
    (one-sided -- decodability only rises above chance). Returns per-bin curves,
    significance masks and thresholds, or None if there are too few trials/sessions.
    """
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return None
    n_groups = len(np.unique(groups))
    if n_groups < 2:
        return None
    folds = int(min(n_splits, n_groups, counts.min()))
    if folds < 2:
        return None
    nBins = X.shape[1]

    def per_bin_curve(y_use):
        # one CV split for this labelling, reused across bins to avoid the
        # per-bin overhead of cross_val_predict.
        cv = StratifiedGroupKFold(n_splits=folds, shuffle=True,
                                  random_state=random_state)
        splits = list(cv.split(X, y_use, groups))
        acc = np.empty(nBins)
        mi = np.empty(nBins)
        for b in range(nBins):
            xb = X[:, [b]]
            y_pred = np.empty_like(y_use)
            for tr, te in splits:
                clf = make_pipeline(
                    StandardScaler(),
                    LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
                clf.fit(xb[tr], y_use[tr])
                y_pred[te] = clf.predict(xb[te])
            acc[b] = balanced_accuracy_score(y_use, y_pred)
            mi[b] = confusion_mi_bits(y_use, y_pred, classes)
        return acc, mi

    obs_acc, obs_mi = per_bin_curve(y)
    perm_rng = np.random.default_rng(random_state)
    max_acc = np.empty(n_perms)
    max_mi = np.empty(n_perms)
    # also accumulate the per-bin null curves so we can plot the empirical
    # (permutation-mean) baseline per time point / frequency, not just a flat
    # theoretical chance line.
    null_acc_sum = np.zeros(nBins)
    null_mi_sum = np.zeros(nBins)
    for i in range(n_perms):
        a, m = per_bin_curve(perm_rng.permutation(y))
        max_acc[i] = np.max(a)
        max_mi[i] = np.max(m)
        null_acc_sum += a
        null_mi_sum += m
    acc_thr = float(np.percentile(max_acc, 100 * (1 - alpha)))
    mi_thr = float(np.percentile(max_mi, 100 * (1 - alpha)))
    acc_null_mean = null_acc_sum / n_perms
    mi_null_mean = null_mi_sum / n_perms
    return {
        'acc': obs_acc, 'mi': obs_mi,
        'acc_sig': obs_acc > acc_thr, 'mi_sig': obs_mi > mi_thr,
        'acc_thr': acc_thr, 'mi_thr': mi_thr,
        'acc_null_mean': acc_null_mean, 'mi_null_mean': mi_null_mean,
        'chance': float(1.0 / len(classes)),
        'n_folds': folds, 'n_groups': int(n_groups),
    }

def run_resolved(X, y, groups, feature, level, name, axis, xlabel):
    """Worker: clean, run resolved_decode, attach metadata. Pure compute -- plotting
    and saving happen in the main process."""
    warnings.filterwarnings(
        "ignore", message="y_pred contains classes not in y_true")
    if X is None:
        return None
    X, y, groups = clean_xy(X, y, groups)
    if X.shape[0] < min_trials_per_state * 2 or len(np.unique(y)) < 2:
        return None
    res = resolved_decode(X, y, groups, n_perms=n_perms, n_splits=n_splits,
                          random_state=RANDOM_STATE, alpha=RESOLVED_ALPHA)
    if res is None:
        return None
    res.update({'feature': feature, 'level': level, 'name': name,
                'axis': np.asarray(axis), 'xlabel': xlabel})
    return res

# -----------------------------
# Decoding analysis
# -----------------------------
print(f"\n=== Decoding state from {FEATURE_SETS} (all sessions pooled) ===")
if not state_data_timelock:
    raise SystemExit("No timelock data collected -- nothing to decode.")

first_channels = state_data_timelock[next(iter(state_data_timelock))][0]['channels']
Sig_CH = np.array_split(first_channels, 6)

# Build the full list of decode jobs. Pooling is cheap and done serially here;
# the expensive permutation tests are dispatched together so all cores stay busy.
# Each feature representation ('erp', 'spectrum') is decoded independently.
jobs = []  # each: (X, y, groups, level, array_index, name, feature)

for feature in FEATURE_SETS:
    # ---------- Per-channel ----------
    for i_arr, ch_names in enumerate(Sig_CH):
        for ch_name in ch_names:
            X, y, groups = pool_channel(state_data_timelock, ch_name, feature)
            jobs.append((X, y, groups, 'channel', i_arr + 1, ch_name, feature))

    # ---------- Per-array (mean feature across channels) ----------
    for i_arr, ch_names in enumerate(Sig_CH):
        X, y, groups = pool_array(state_data_timelock, ch_names, feature)
        jobs.append((X, y, groups, 'array', i_arr + 1, f'array{i_arr+1}', feature))

    # ---------- Combined arrays (1-3 merged, 4/5/6 separate) ----------
    for i_arr, ch_names in enumerate(Sig_CH):
        if i_arr < 3:
            if i_arr == 0:
                combined_ch_names = np.concatenate(Sig_CH[:3])
                name = 'Array13'
            else:
                continue
        else:
            combined_ch_names = ch_names
            name = f'Array{i_arr+1}'
        X, y, groups = pool_array(state_data_timelock, combined_ch_names, feature)
        jobs.append((X, y, groups, 'combined_array', i_arr + 1, name, feature))

print(f"  --> running {len(jobs)} decode jobs across {N_JOBS} workers")
results = Parallel(n_jobs=N_JOBS, verbose=5)(
    delayed(decode_job)(*job) for job in jobs
)
summary_rows = [r for r in results if r is not None]

# -----------------------------
# Save summary + plots
# -----------------------------
if not summary_rows:
    raise SystemExit("No decoding results produced.")

summary_df = pd.DataFrame(summary_rows)
# Report accuracy relative to the EMPIRICAL baseline (permutation-null mean) rather
# than the theoretical 1/nStates: under session-grouped CV with imbalanced states
# the null mean sits above 1/nStates, so this margin is the honest effect size.
summary_df['acc_above_null'] = (
    summary_df['balanced_accuracy'] - summary_df['perm_mean']
)
# Benjamini-Hochberg FDR within each (feature, level) family (channel / array /
# combined_array): the per-channel family is ~160 tests, so a raw p<0.05 map
# would carry ~8 false positives. Correct each family separately, per feature.
summary_df['p_value_fdr'] = (
    summary_df.groupby(['feature', 'level'])['p_value'].transform(bh_fdr)
)
summary_df['mi_p_value_fdr'] = (
    summary_df.groupby(['feature', 'level'])['mi_p_value'].transform(bh_fdr)
)
summary_csv = os.path.join(results_data_dir, "decoding_summary.csv")
summary_df.to_csv(summary_csv, index=False)
print(f"\nWrote decoding summary: {summary_csv}")

# Plots are produced per feature representation; filenames carry the feature tag.
for feature in FEATURE_SETS:
    feat_df = summary_df[summary_df['feature'] == feature]
    if feat_df.empty:
        continue

    # ---- Per-channel accuracy bar plots, grouped by array ----
    chan_df = feat_df[feat_df['level'] == 'channel']
    for i_arr in sorted(chan_df['array_index'].unique()):
        sub = chan_df[chan_df['array_index'] == i_arr]
        fig, ax = plt.subplots(figsize=(max(6, 0.4 * len(sub)), 4))
        x = np.arange(len(sub))
        sig = sub['p_value_fdr'].to_numpy() < 0.05
        colors = ['tab:green' if s else 'tab:gray' for s in sig]
        ax.bar(x, sub['balanced_accuracy'], color=colors)
        # Empirical baseline = permutation-null mean (per bar): the honest chance
        # level under session-grouped CV, which sits above the theoretical 1/nStates.
        for xi, (_, row) in zip(x, sub.iterrows()):
            ax.hlines(row['perm_mean'], xi - 0.4, xi + 0.4, color='red', lw=1.2,
                      label='perm. mean' if xi == 0 else None)
        ax.axhline(sub['chance'].iloc[0], color='gray', ls=':', lw=0.8,
                   label='1/nStates (theoretical)')
        ax.set_xticks(x)
        ax.set_xticklabels(sub['name'], rotation=90, fontsize=6)
        ax.set_ylabel('Balanced accuracy')
        ax.set_title(f'State decoding [{feature}] - Array {i_arr} (green: FDR q<0.05)')
        ax.legend(fontsize=7)
        plt.tight_layout()
        fig.savefig(os.path.join(
            output_dir, f'decoding_channels_{feature}_array{i_arr}.pdf'))
        plt.close(fig)

    # ---- Array-level + combined-array accuracy and MI plots ----
    arr_df = feat_df[feat_df['level'].isin(['array', 'combined_array'])]
    if not arr_df.empty:
        x = np.arange(len(arr_df))
        xticklabels = arr_df['level'] + ':' + arr_df['name']

        # accuracy vs the empirical (permutation-mean) baseline
        fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(arr_df)), 4))
        sig = arr_df['p_value_fdr'].to_numpy() < 0.05
        colors = ['tab:green' if s else 'tab:gray' for s in sig]
        ax.bar(x, arr_df['balanced_accuracy'], color=colors)
        # Red = permutation-null mean per bar (the honest chance under grouped CV);
        # faint gray = theoretical 1/nStates, shown only for reference.
        for xi, (_, row) in zip(x, arr_df.iterrows()):
            ax.hlines(row['perm_mean'], xi - 0.4, xi + 0.4, color='red', lw=1.2,
                      label='perm. mean' if xi == 0 else None)
        ax.axhline(arr_df['chance'].iloc[0], color='gray', ls=':', lw=0.8,
                   label='1/nStates (theoretical)')
        ax.set_xticks(x)
        ax.set_xticklabels(xticklabels, rotation=90, fontsize=7)
        ax.set_ylabel('Balanced accuracy')
        ax.set_title(f'State decoding [{feature}] - arrays '
                     f'(green: FDR q<0.05 vs perm. null)')
        ax.legend(fontsize=7)
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f'decoding_arrays_{feature}.pdf'))
        plt.close(fig)

        # bias-corrected mutual information as % of state entropy
        fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(arr_df)), 4))
        sig_mi = arr_df['mi_p_value_fdr'].to_numpy() < 0.05
        colors = ['tab:blue' if s else 'tab:gray' for s in sig_mi]
        ax.bar(x, 100 * arr_df['mi_frac_entropy'], color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(xticklabels, rotation=90, fontsize=7)
        ax.set_ylabel('Information about state (% of H[state])')
        ax.set_title(f'State information in LFP [{feature}] (blue: FDR q<0.05)')
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f'mi_arrays_{feature}.pdf'))
        plt.close(fig)

# LDA decoding pattern for one channel (which part of the ERP carries state info).
# Only meaningful for the time-domain 'erp' feature (pattern is over time).
if 'erp' in FEATURE_SETS:
    erp_chan_df = summary_df[(summary_df['feature'] == 'erp') &
                             (summary_df['level'] == 'channel')]
    time_vec = state_data_timelock[next(iter(state_data_timelock))][0]['x']
    if PATTERN_CHANNEL is not None:
        pat_ch = PATTERN_CHANNEL
    elif not erp_chan_df.empty:
        pat_ch = erp_chan_df.loc[erp_chan_df['p_value'].idxmin(), 'name']  # best channel
    else:
        pat_ch = None

    if pat_ch is not None:
        Xp, yp, gp = pool_channel(state_data_timelock, pat_ch, 'erp')
        if Xp is not None:
            Xp, yp, gp = clean_xy(Xp, yp, gp)
            classes, patterns = lda_activation_patterns(Xp, yp)
            fig, ax = plt.subplots(figsize=(7, 4))
            for cls, pat in zip(classes, patterns):
                ax.plot(time_vec, pat, label=f'state {int(cls)}')
            ax.axvline(0, color='k', lw=0.8, ls=':', label='stim onset')
            ax.axhline(0, color='gray', lw=0.6)
            ax.set_xlabel('Time from stim onset (s)')
            ax.set_ylabel('Activation pattern (LFP units)')
            ax.set_title(f'LDA decoding pattern - channel {pat_ch} (Haufe-transformed)')
            ax.legend(fontsize=7)
            plt.tight_layout()
            fig.savefig(os.path.join(output_dir, f'lda_pattern_{pat_ch}.pdf'))
            plt.close(fig)
            print(f"Wrote LDA decoding pattern for channel {pat_ch}")

# -----------------------------
# Time/frequency-resolved decoding (array level): localise WHICH time points
# ('erp') and frequencies ('spectrum') carry state information.
# -----------------------------
if RESOLVED_FEATURES:
    print("\n=== Resolved (per-bin) decoding to localise state information ===")
    first_entry = state_data_timelock[next(iter(state_data_timelock))][0]
    axis_for = {'erp': first_entry['x'], 'spectrum': first_entry['freqs']}
    xlabel_for = {'erp': 'Time from stim onset (s)', 'spectrum': 'Frequency (Hz)'}

    resolved_jobs = []  # (X, y, groups, feature, level, name, axis, xlabel)
    for feature in RESOLVED_FEATURES:
        axis = axis_for[feature]
        xlabel = xlabel_for[feature]
        # arrays
        for i_arr, ch_names in enumerate(Sig_CH):
            X, y, groups = pool_array(state_data_timelock, ch_names, feature)
            resolved_jobs.append((X, y, groups, feature, 'array',
                                  f'array{i_arr+1}', axis, xlabel))
        # combined arrays (1-3 merged, 4/5/6 separate)
        for i_arr, ch_names in enumerate(Sig_CH):
            if i_arr < 3:
                if i_arr == 0:
                    combined_ch_names = np.concatenate(Sig_CH[:3])
                    name = 'Array13'
                else:
                    continue
            else:
                combined_ch_names = ch_names
                name = f'Array{i_arr+1}'
            X, y, groups = pool_array(state_data_timelock, combined_ch_names, feature)
            resolved_jobs.append((X, y, groups, feature, 'combined_array',
                                  name, axis, xlabel))

    print(f"  --> running {len(resolved_jobs)} resolved jobs across {N_JOBS} workers")
    resolved_results = Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(run_resolved)(*job) for job in resolved_jobs
    )
    resolved_results = [r for r in resolved_results if r is not None]

    for res in resolved_results:
        axis = res['axis']
        feat, level, name = res['feature'], res['level'], res['name']

        npz_path = os.path.join(
            results_data_dir, f"resolved_{feat}_{level}_{name}.npz")
        np.savez_compressed(
            npz_path, axis=axis, acc=res['acc'], mi=res['mi'],
            acc_sig=res['acc_sig'], mi_sig=res['mi_sig'],
            acc_thr=res['acc_thr'], mi_thr=res['mi_thr'],
            acc_null_mean=res['acc_null_mean'], mi_null_mean=res['mi_null_mean'],
            chance=res['chance'], feature=feat, level=level, name=name)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
        # Red = per-bin permutation-null mean (empirical baseline); faint gray =
        # theoretical 1/nStates for reference; green dotted = max-stat threshold.
        ax1.plot(axis, res['acc'], color='k')
        ax1.plot(axis, res['acc_null_mean'], color='red', lw=1, label='perm. mean')
        ax1.axhline(res['chance'], color='gray', ls=':', lw=0.8,
                    label='1/nStates (theoretical)')
        ax1.axhline(res['acc_thr'], color='tab:green', ls=':', lw=1,
                    label=f"max-stat thr (a={RESOLVED_ALPHA})")
        ax1.fill_between(axis, res['acc'], res['acc_null_mean'], where=res['acc_sig'],
                         color='tab:green', alpha=0.3)
        ax2.plot(axis, res['mi'], color='k')
        ax2.plot(axis, res['mi_null_mean'], color='red', lw=1, label='perm. mean')
        ax2.axhline(res['mi_thr'], color='tab:blue', ls=':', lw=1, label='max-stat thr')
        ax2.fill_between(axis, res['mi'], res['mi_null_mean'], where=res['mi_sig'],
                         color='tab:blue', alpha=0.3)
        if feat == 'erp':
            ax1.axvline(0, color='gray', lw=0.8, ls=':')
            ax2.axvline(0, color='gray', lw=0.8, ls=':')
        ax1.set_ylabel('Balanced accuracy')
        ax1.legend(fontsize=7)
        ax1.set_title(f"Resolved decoding [{feat}] - {level}:{name} "
                      f"(shaded: max-stat q<{RESOLVED_ALPHA})")
        ax2.set_ylabel('MI (bits)')
        ax2.set_xlabel(res['xlabel'])
        ax2.legend(fontsize=7)
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f"resolved_{feat}_{level}_{name}.pdf"))
        plt.close(fig)
    print(f"  --> wrote {len(resolved_results)} resolved decoding figures")

print(f"Plots saved under {output_dir}")
print(f"Results saved under {results_data_dir}")
