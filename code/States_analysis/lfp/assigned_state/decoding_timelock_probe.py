"""
Summary
-------
Probe whether the *time-domain* (timelock) LFP carries single-trial information
about cognitive state, and whether the linear decoder was the limiting factor.

Motivation
----------
`state_lfp_decoding.py` decodes state from the raw stim-aligned ERP with a linear
LDA and finds ~0 bits / chance accuracy for states [0, 2], while the power
*spectrum* decodes well (~44-67 Hz). Two confounds could hide a real time-domain
effect:
  1. the MODEL: LDA reads phase-locked amplitude only and cannot represent power
     (a second-order / variance feature), so a real effect could be invisible to
     it but visible to a nonlinear learner;
  2. the REPRESENTATION: the state difference may live in non-phase-locked
     (induced) power, which never appears in the trial-mean ERP amplitude but is
     present in the single-trial variance / band envelope.

This script disentangles the two by crossing three time-domain feature sets with
a linear and a nonlinear model, under the same session-grouped CV + label-shuffle
null as the main decoder:

  features (all derived from the timelock window):
    * erp_raw   -- stim-aligned amplitude, decimated to TARGET_FS (phase-locked
                   evoked shape; identical representation to state_lfp_decoding.py)
    * variance  -- per-trial log-variance in sub-windows of length WIN_MS
                   (non-phase-locked / induced power, in the time domain)
    * gamma_env -- Hilbert envelope in GAMMA_BAND, mean per sub-window (the band
                   the spectrum flagged)

  models:
    * LDA  -- shrinkage LDA (linear), matching the main decoder
    * HGB  -- HistGradientBoosting (nonlinear)

Reading the result:
  - erp_raw stays at chance for BOTH models   -> the evoked SHAPE carries no
    single-trial state info (and the linear model was not the bottleneck there).
  - variance / gamma_env decode above chance   -> the state difference is induced
    power, not evoked shape -- consistent with the spectrum result.

Outputs (array level), every filename and plot title tagged with the states used
and the model:
  - timelock_probe_summary_states<S>.csv  -- one row per feature/model/array.
  - decode_states<S>_<feature>_<model>_arrays.pdf -- balanced accuracy vs the
    permutation-null baseline, per array.
  - mi_states<S>_<feature>_<model>_arrays.pdf -- bias-corrected MI (% of H[state]).
  - overview_states<S>_accuracy.pdf / overview_states<S>_mi.pdf -- feature x model
    grid so the linear-vs-nonlinear and shape-vs-power contrasts are read at a glance.
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
from scipy.signal import decimate, butter, sosfiltfilt, hilbert
from joblib import Parallel, delayed
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis)
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

warnings.filterwarnings(
    "ignore", message="y_pred contains classes not in y_true")

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
# Output roots; the state-tagged leaf (e.g. ".../timelock_probe/states_0_2") is
# appended below so the path itself records which states were compared.
output_root = f'/cs/projects/MWzeronoise/Analysis/4Shivangi/plots/states_lfp/all_trials/{WIN_TAG}/decoding/timelock_probe'
results_dir = '/mnt/cs/projects/MWzeronoise/Analysis/4Shivangi/Results/states_analysis/states_lfp'
results_root = os.path.join(results_dir, "all_trials", WIN_TAG, "decoding", "timelock_probe")

sessions = ['20230203', '20230208', '20230209', '20230213', '20230214']
STATES_TO_DECODE = [0, 2]          # states compared; tags every output
STATE_TAG = '_'.join(str(s) for s in STATES_TO_DECODE)  # e.g. "0_2"

output_dir = os.path.join(output_root, f'states_{STATE_TAG}')
results_data_dir = os.path.join(results_root, f'states_{STATE_TAG}')
os.makedirs(output_dir, exist_ok=True)
os.makedirs(results_data_dir, exist_ok=True)

FEATURE_SETS = ['erp_raw', 'variance', 'gamma_env']
# LDA: linear baseline (mean-amplitude differences only).
# QDA: hypothesis-matched -- class-specific covariance -> quadratic boundary, so
#      it reads second-order (variance/power) structure that LDA's shared-
#      covariance assumption discards. On erp_raw this is the direct test of
#      "is the time-domain state difference a power effect hidden in amplitude?".
# HGB: fast catch-all nonlinear learner (arbitrary interactions), as insurance.
MODELS = ['LDA', 'QDA', 'HGB']
QDA_REG = 0.3                      # covariance shrinkage for QDA (reg_param)
QDA_PCA_VAR = 0.95                 # PCA variance kept before QDA, to keep the
                                   # per-class covariance well-conditioned

WIN_MS = 50                        # sub-window length (ms) for variance / envelope
GAMMA_BAND = [40, 70]             # Hz, band for the Hilbert envelope feature
n_perms = 200                      # label shuffles for the null / p-value
n_splits = 5                       # stratified CV folds
N_JOBS = -1
ORIG_FS = 1000
TARGET_FS = 200                    # erp_raw decimation target
DECIM_Q = ORIG_FS // TARGET_FS
min_trials_per_state = 5
RANDOM_STATE = 42

# load states info for all sessions
state_probs = np.load(f'{states_data_dir}/foraging_shivangi_no_sess1_clipped_state_assignments.npy')
with open(f'{states_data_dir}/foraging_shivangi_no_sess1_clipped_session_index.json') as f:
    session_index = json.load(f)

session_to_probs = {}
for sess in session_index:
    session_date = sess['session_id'].split('_')[1]
    session_to_probs[session_date] = state_probs[sess['start_idx']: sess['end_idx'] + 1]


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
            (datalfp.trialdefinition, np.arange(nTrials).reshape(-1, 1)))


def subwindow_edges(time_vec, win_ms):
    """Indices that split the time axis into consecutive ~win_ms sub-windows."""
    win_s = win_ms / 1000.0
    dt = np.median(np.diff(time_vec))
    step = max(1, int(round(win_s / dt)))
    return list(range(0, len(time_vec) + 1, step)), step


def trial_variance_features(trials_array, time_vec, win_ms):
    """(nTrials, nTime, nCh) -> (nTrials, nBins, nCh) log-variance per sub-window.
    Variance is non-phase-locked power: it survives even when the trial-mean ERP
    is flat, so it reads induced power directly from the time domain."""
    edges, _ = subwindow_edges(time_vec, win_ms)
    feats = []
    for a, b in zip(edges[:-1], edges[1:]):
        if b - a < 2:
            continue
        feats.append(np.var(trials_array[:, a:b, :], axis=1))  # (nTrials, nCh)
    feats = np.stack(feats, axis=1)  # (nTrials, nBins, nCh)
    return np.log10(feats + 1e-20)


def trial_envelope_features(trials_array, time_vec, band, fs, win_ms):
    """(nTrials, nTime, nCh) -> (nTrials, nBins, nCh) mean log Hilbert-envelope
    per sub-window in `band`. Band-limited amplitude, also non-phase-locked."""
    sos = butter(4, band, btype='bandpass', fs=fs, output='sos')
    filt = sosfiltfilt(sos, trials_array, axis=1)
    env = np.abs(hilbert(filt, axis=1))      # instantaneous amplitude
    edges, _ = subwindow_edges(time_vec, win_ms)
    feats = []
    for a, b in zip(edges[:-1], edges[1:]):
        if b - a < 2:
            continue
        feats.append(np.mean(env[:, a:b, :], axis=1))  # (nTrials, nCh)
    feats = np.stack(feats, axis=1)
    return np.log10(feats + 1e-20)


def state_entropy_bits(y):
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log2(p)))


def confusion_mi_bits(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels).astype(float)
    total = cm.sum()
    if total == 0:
        return 0.0
    p_joint = cm / total
    p_true = p_joint.sum(axis=1, keepdims=True)
    p_pred = p_joint.sum(axis=0, keepdims=True)
    p_indep = p_true @ p_pred
    nz = (p_joint > 0) & (p_indep > 0)
    return float(np.sum(p_joint[nz] * np.log2(p_joint[nz] / p_indep[nz])))


def make_clf(model):
    """Pipeline factory.
    LDA -- linear, shrinkage covariance; matches the main decoder.
    QDA -- per-class covariance (quadratic boundary), the targeted second-order
           test. PCA first keeps the per-class covariance well-conditioned when
           there are many time points (e.g. ~220 for erp_raw), reg_param shrinks
           what remains.
    HGB -- nonlinear gradient-boosted trees; scale-invariant, no scaler needed."""
    if model == 'LDA':
        return make_pipeline(
            StandardScaler(),
            LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
    if model == 'QDA':
        return make_pipeline(
            StandardScaler(),
            PCA(n_components=QDA_PCA_VAR, random_state=RANDOM_STATE),
            QuadraticDiscriminantAnalysis(reg_param=QDA_REG))
    if model == 'HGB':
        return HistGradientBoostingClassifier(
            max_depth=3, max_iter=200, learning_rate=0.05,
            l2_regularization=1.0, random_state=RANDOM_STATE)
    raise ValueError(model)


def decode_state(X, y, groups, model, n_perms=200, n_splits=5, random_state=42):
    """Session-grouped CV decoding with a label-shuffle null. Returns balanced
    accuracy, bias-corrected confusion-MI, and permutation p-values, or None."""
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return None
    n_groups = len(np.unique(groups))
    if n_groups < 2:
        return None
    folds = int(min(n_splits, n_groups, counts.min()))
    if folds < 2:
        return None

    def cv_metrics(y_use):
        cv = StratifiedGroupKFold(n_splits=folds, shuffle=True,
                                  random_state=random_state)
        y_pred = cross_val_predict(make_clf(model), X, y_use, groups=groups,
                                   cv=cv, n_jobs=1)
        return (balanced_accuracy_score(y_use, y_pred),
                confusion_mi_bits(y_use, y_pred, classes))

    acc_obs, mi_obs = cv_metrics(y)
    perm_rng = np.random.default_rng(random_state)
    perm_acc = np.empty(n_perms)
    perm_mi = np.empty(n_perms)
    for i in range(n_perms):
        perm_acc[i], perm_mi[i] = cv_metrics(perm_rng.permutation(y))

    p_acc = (1 + np.sum(perm_acc >= acc_obs)) / (n_perms + 1)
    p_mi = (1 + np.sum(perm_mi >= mi_obs)) / (n_perms + 1)
    mi_corrected = max(0.0, mi_obs - float(np.mean(perm_mi)))
    H_state = state_entropy_bits(y)
    return {
        'balanced_accuracy': float(acc_obs), 'chance': float(1.0 / len(classes)),
        'p_value': float(p_acc), 'perm_mean': float(np.mean(perm_acc)),
        'mi_bits_raw': float(mi_obs), 'mi_bits': float(mi_corrected),
        'mi_null_mean': float(np.mean(perm_mi)),
        'state_entropy_bits': float(H_state),
        'mi_frac_entropy': float(mi_corrected / H_state) if H_state > 0 else 0.0,
        'mi_p_value': float(p_mi), 'n_folds': folds, 'n_groups': int(n_groups),
    }


# -----------------------------
# Data collection
# -----------------------------
# state -> list (per session) of {'erp_raw','variance','gamma_env': (nTr,nBins,nCh),
#                                  'channels', 'session'}
store = {}

for session_name in sessions:
    print(f"\n=== Processing session {session_name} ===")
    lfp_path = os.path.join(lfp_data_dir, session_name, 'Cleaned_lfp_FT.spy')
    trial_info_path = os.path.join(trial_info_dir, session_name, 'Trial_Info.pkl')
    if not os.path.exists(lfp_path) or not os.path.exists(trial_info_path):
        continue

    predicted_states = session_to_probs[session_name]
    trial_info_df = pd.read_pickle(trial_info_path)
    trial_info_df.iloc[:, 0] = (trial_info_df.iloc[:, 0] - 1000).astype('Int64')
    stim_df = pd.DataFrame({'TrialIndex': np.arange(len(predicted_states)),
                            'States': predicted_states})
    combined_df = pd.merge(trial_info_df, stim_df, left_on='Trial_Number',
                           right_on='TrialIndex', how='inner')

    datalfp = spy.load(lfp_path)
    ensure_trialindex_in_trialdefinition(datalfp)
    cfg = spy.StructDict(latency=LATENCY)
    data = spy.selectdata(cfg, datalfp)
    selected_trials = data.trialdefinition[:, 3].astype(int)
    states_trial_info_filt = combined_df[combined_df['TrialIndex'].isin(selected_trials)]
    available_states = np.sort(np.unique(states_trial_info_filt['States'].to_numpy()))
    unique_states = np.array([s for s in available_states if s in STATES_TO_DECODE])

    for state_value in unique_states:
        cfg_sel = spy.StructDict(
            trials=np.where(states_trial_info_filt['States'] == state_value)[0])
        datas_state = spy.selectdata(cfg_sel, data)
        datas_state_clean, valid_channels = remove_nan_trials_channels(datas_state)
        if datas_state_clean is None:
            continue
        trials_array = np.stack(datas_state_clean.trials, axis=0)  # (nTr, nTime, nCh)
        time_vec = np.asarray(datas_state_clean.time[0])
        if trials_array.shape[0] < min_trials_per_state:
            continue

        # erp_raw: decimated stim-aligned amplitude (phase-locked shape).
        if DECIM_Q > 1:
            erp_raw = decimate(trials_array, DECIM_Q, axis=1, ftype='fir')
        else:
            erp_raw = trials_array
        # induced-power features from the same timelock window.
        var_feat = trial_variance_features(trials_array, time_vec, WIN_MS)
        env_feat = trial_envelope_features(trials_array, time_vec, GAMMA_BAND,
                                           ORIG_FS, WIN_MS)

        store.setdefault(state_value, []).append(
            {'erp_raw': erp_raw, 'variance': var_feat, 'gamma_env': env_feat,
             'channels': valid_channels, 'session': session_name})


# -----------------------------
# Pooling + jobs
# -----------------------------
def pool_array(store, ch_group, feature):
    """Array-level per-trial vector = mean feature across the array's channels."""
    X_parts, y_parts, g_parts = [], [], []
    for state_value in sorted(store.keys()):
        for sess in store[state_value]:
            ch_valid = [c for c in ch_group if c in sess['channels']]
            if not ch_valid:
                continue
            ch_idx = [sess['channels'].index(c) for c in ch_valid]
            n = sess[feature].shape[0]
            X_parts.append(np.mean(sess[feature][:, :, ch_idx], axis=2))
            y_parts.append(np.full(n, state_value))
            g_parts.append(np.full(n, sess['session']))
    if not X_parts:
        return None, None, None
    return (np.concatenate(X_parts, axis=0), np.concatenate(y_parts),
            np.concatenate(g_parts))


def clean_xy(X, y, groups):
    good = ~np.any(~np.isfinite(X), axis=1)
    return X[good], y[good], groups[good]


def decode_job(X, y, groups, level, name, feature, model):
    warnings.filterwarnings(
        "ignore", message="y_pred contains classes not in y_true")
    if X is None:
        return None
    X, y, groups = clean_xy(X, y, groups)
    if X.shape[0] < min_trials_per_state * 2 or len(np.unique(y)) < 2:
        return None
    res = decode_state(X, y, groups, model, n_perms=n_perms, n_splits=n_splits,
                       random_state=RANDOM_STATE)
    if res is None:
        return None
    return {'feature': feature, 'model': model, 'level': level, 'name': name,
            'n_trials': int(X.shape[0]), 'n_features': int(X.shape[1]),
            'n_states': int(len(np.unique(y))), 'n_sessions': res['n_groups'],
            **{k: res[k] for k in (
                'balanced_accuracy', 'chance', 'perm_mean', 'p_value',
                'mi_bits', 'mi_bits_raw', 'mi_null_mean', 'state_entropy_bits',
                'mi_frac_entropy', 'mi_p_value', 'n_folds')}}


if not store:
    raise SystemExit("No data collected -- nothing to decode.")

first_channels = store[next(iter(store))][0]['channels']
Sig_CH = np.array_split(first_channels, 6)

jobs = []
for feature in FEATURE_SETS:
    for model in MODELS:
        # arrays 1-6
        for i_arr, ch_names in enumerate(Sig_CH):
            X, y, g = pool_array(store, ch_names, feature)
            jobs.append((X, y, g, 'array', f'array{i_arr+1}', feature, model))
        # combined arrays (1-3 merged, 4/5/6 separate)
        for i_arr, ch_names in enumerate(Sig_CH):
            if i_arr < 3:
                if i_arr == 0:
                    comb, name = np.concatenate(Sig_CH[:3]), 'Array13'
                else:
                    continue
            else:
                comb, name = ch_names, f'Array{i_arr+1}'
            X, y, g = pool_array(store, comb, feature)
            jobs.append((X, y, g, 'combined_array', name, feature, model))

print(f"\nStates {STATES_TO_DECODE} | features {FEATURE_SETS} | models {MODELS}")
print(f"  --> running {len(jobs)} decode jobs across {N_JOBS} workers")
results = Parallel(n_jobs=N_JOBS, verbose=5)(delayed(decode_job)(*j) for j in jobs)
summary_rows = [r for r in results if r is not None]
if not summary_rows:
    raise SystemExit("No decoding results produced.")

summary_df = pd.DataFrame(summary_rows)
summary_df.insert(0, 'states', STATE_TAG)
summary_df['acc_above_null'] = summary_df['balanced_accuracy'] - summary_df['perm_mean']
summary_csv = os.path.join(results_data_dir, "timelock_probe_summary.csv")
summary_df.to_csv(summary_csv, index=False)
print(f"\nWrote summary: {summary_csv}")


# -----------------------------
# Plots (every title/filename tagged with states + model)
# -----------------------------
def bar_plot(sub, value_col, ylabel, title, fname, baseline_col=None,
             chance=None, sig_col=None):
    sub = sub.reset_index(drop=True)
    x = np.arange(len(sub))
    sig = (sub[sig_col].to_numpy() < 0.05) if sig_col else np.zeros(len(sub), bool)
    colors = ['tab:green' if s else 'tab:gray' for s in sig]
    fig, ax = plt.subplots(figsize=(max(6, 0.7 * len(sub)), 4))
    ax.bar(x, sub[value_col], color=colors)
    if baseline_col is not None:
        for xi, (_, row) in zip(x, sub.iterrows()):
            ax.hlines(row[baseline_col], xi - 0.4, xi + 0.4, color='red', lw=1.2,
                      label='perm. mean' if xi == 0 else None)
    if chance is not None:
        ax.axhline(chance, color='gray', ls=':', lw=0.8, label='1/nStates')
    ax.set_xticks(x)
    ax.set_xticklabels(sub['level'] + ':' + sub['name'], rotation=90, fontsize=7)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=7)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, fname))
    plt.close(fig)


for feature in FEATURE_SETS:
    for model in MODELS:
        sub = summary_df[(summary_df['feature'] == feature) &
                         (summary_df['model'] == model)]
        if sub.empty:
            continue
        # States live in the folder name; titles still note them for figures
        # viewed in isolation, but filenames carry only feature/model.
        tag = f"states {STATE_TAG} | feature={feature} | model={model}"
        bar_plot(sub, 'balanced_accuracy', 'Balanced accuracy',
                 f'State decoding [{tag}]\n(green: p<0.05 vs perm. null)',
                 f'decode_{feature}_{model}_arrays.pdf',
                 baseline_col='perm_mean', chance=sub['chance'].iloc[0],
                 sig_col='p_value')
        bar_plot(sub, 'mi_frac_entropy', 'MI (fraction of H[state])',
                 f'State information [{tag}]\n(green: MI p<0.05)',
                 f'mi_{feature}_{model}_arrays.pdf',
                 sig_col='mi_p_value')

# Overview grids: feature (rows) x model (cols), one bar per array. Lets you read
# the linear-vs-nonlinear and shape-vs-power contrasts at a glance.
def overview(value_col, baseline_col, ylabel, title_metric, fname):
    nF, nM = len(FEATURE_SETS), len(MODELS)
    fig, axes = plt.subplots(nF, nM, figsize=(4.5 * nM, 3 * nF),
                             sharey=True, squeeze=False)
    for r, feature in enumerate(FEATURE_SETS):
        for c, model in enumerate(MODELS):
            ax = axes[r][c]
            sub = summary_df[(summary_df['feature'] == feature) &
                             (summary_df['model'] == model)].reset_index(drop=True)
            if sub.empty:
                ax.set_visible(False)
                continue
            x = np.arange(len(sub))
            sig = sub['p_value'].to_numpy() < 0.05
            ax.bar(x, sub[value_col],
                   color=['tab:green' if s else 'tab:gray' for s in sig])
            if baseline_col is not None:
                ax.plot(x, sub[baseline_col], 'r_', ms=14, label='perm. mean')
            ax.set_xticks(x)
            ax.set_xticklabels(sub['name'], rotation=90, fontsize=6)
            ax.set_title(f'{feature} | {model}', fontsize=9)
            if c == 0:
                ax.set_ylabel(ylabel, fontsize=8)
    fig.suptitle(f'States {STATE_TAG} -- {title_metric} '
                 f'(green: p<0.05 vs perm. null)', fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(output_dir, fname))
    plt.close(fig)


overview('balanced_accuracy', 'perm_mean', 'Balanced acc', 'decoding accuracy',
         'overview_accuracy.pdf')
overview('mi_frac_entropy', None, 'MI (frac H)', 'state information (MI)',
         'overview_mi.pdf')

# Console summary
print("\n=== Summary (array level, best array per feature/model) ===")
pd.set_option('display.width', 200)
best = (summary_df.sort_values('balanced_accuracy', ascending=False)
        .groupby(['feature', 'model']).head(1)
        .sort_values(['feature', 'model']))
print(best[['feature', 'model', 'name', 'n_features', 'balanced_accuracy',
            'acc_above_null', 'p_value', 'mi_frac_entropy', 'mi_p_value']]
      .to_string(index=False))
print(f"\nPlots saved under {output_dir}")
print(f"Results saved under {results_data_dir}")
