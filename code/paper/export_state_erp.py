#!/usr/bin/env python
"""
=============================================================================================
export_state_erp.py -- event-locked LFP of the two attentional states, per array, with stats
=============================================================================================
Writes writing/figures/data/state_erp.npz, which figures.py fig06 draws.

The inattentive (0) and attentive (2) states are compared at four events:
    stimulus         0.2 s before to 0.9 s after stimulus onset       (as erp_spectra_stats.py)
    reaction         0.45 s before to 0.45 s after the reaction time  (as erp_spectra_stats_RT.py)
    saccade_onset    0.2 s before to 0.3 s after saccade onset        (as saccade_triggered_average.py)
    saccade_offset   0.2 s before to 0.3 s after saccade offset (landing)

Per session, state, event and channel the epochs are averaged (an epoch enters a channel's average
only if it has no missing sample on that channel). A saccade takes the state of the trial it falls
in. The session trace of an array is the mean over its channels, and the arrays are the recorded
arrays: channel_001-032 is array 1, and so on. (The earlier state plots split the surviving channels
into six equal groups instead.) The two states are compared per array with the session-label
permutation test of erp_spectra_stats.py: 1,000 reassignments of the state labels of the session
traces, maximum and minimum of the difference across the window, two-sided alpha 0.05.

No baseline correction, as in the stimulus- and reaction-locked scripts. Microsaccades (< 6 ms or
< 1 deg) and saccades followed by another within 300 ms are excluded, as in the saccade script.

Run (warping env; about an hour, reads the full-length LFP of five sessions):
    cd /mnt/cs/projects/MWzeronoise/Analysis/4Shivangi/code/paper
    /gs/home/patels/.conda/envs/warping/bin/python export_state_erp.py
then  figures.py fig06  and  ../../writing/latex/build.py
"""

import os
import sys
import glob
import json

import numpy as np
import pandas as pd
import syncopy as spy

ROOT = '/mnt/cs/projects/MWzeronoise/Analysis/4Shivangi'
sys.path.insert(1, os.path.join(ROOT, 'code', 'functions'))
sys.path.insert(1, os.path.join(ROOT, 'code', 'functions', 'eyetracking'))
sys.path.insert(1, os.path.join(ROOT, 'code', 'functions', 'unreal_logfile'))
import time_conversion as tc                 # noqa: E402
from parse_logfile import TextLog            # noqa: E402

# ── Settings ──────────────────────────────────────────────────────────────────────────────
LFP_DIR = os.path.join(ROOT, 'Datasets', 'neural_data', 'stimAalign_cut', 'clean_full_length')
TRIAL_INFO_DIR = os.path.join(ROOT, 'Datasets', 'neural_data', 'stimAalign_cut', 'full_length')
STATES_DIR = os.path.join(ROOT, 'Datasets', 'states_analysis')
EYE_DIR = os.path.join(ROOT, 'Datasets', 'eye_data')
SACC_NPZ = os.path.join(ROOT, 'Results', 'saccade_detection', 'stitched_sessions.npz')
OUT = os.path.join(ROOT, 'writing', 'figures', 'data', 'state_erp.npz')

SESSIONS = ['20230203', '20230208', '20230209', '20230213', '20230214']
SESSION_FOLDERS = {
    '20230203': 'Cosmos_20230203_LeafForaging_001', '20230208': 'Cosmos_20230208_LeafForaging_001',
    '20230209': 'Cosmos_20230209_LeafForaging_001', '20230213': 'Cosmos_20230213_LeafForaging_002',
    '20230214': 'Cosmos_20230214_LeafForaging_001',
}
SESSION_LOGS = {
    '20230203': '2023_02_03-11_35_57_Cosmos_LeafForaging_001_MS_GrassyLandscapeWithBackgroundDark_Cont.log',
    '20230208': '2023_02_08-10_58_17_Cosmos_LeafForaging_001_MS_GrassyLandscapeWithBackgroundDark_Cont.log',
    '20230209': '2023_02_09-11_19_51_Cosmos_LeafForaging_001_KAS_GrassyLandscapeWithBackgroundDark_Cont.log',
    '20230213': '2023_02_13-11_13_43_Cosmos_LeafForaging_002_MS_GrassyLandscapeWithBackgroundDark_Cont.log',
    '20230214': '2023_02_14-11_42_27_Cosmos_LeafForaging_001_PAF_GrassyLandscapeWithBackgroundDark_Cont.log',
}

STATES = (0, 2)                               # inattentive, attentive
EVENTS = {'stimulus': (0.2, 0.9), 'reaction': (0.45, 0.45),
          'saccade_onset': (0.2, 0.3), 'saccade_offset': (0.2, 0.3)}     # (pre, post) in s
N_CH, CH_PER_ARRAY, N_ARRAYS = 192, 32, 6
MIN_SACC_DUR_MS, MICRO_AMP_DEG, NEXT_SACC_GAP_MS = 6.0, 1.0, 300
N_PERMS, ALPHA = 1000, 0.05
rng = np.random.default_rng(42)


# ── Trials: state and reaction time ───────────────────────────────────────────────────────
def session_states():
    probs = np.load(os.path.join(STATES_DIR, 'foraging_shivangi_no_sess1_clipped_state_assignments.npy'))
    with open(os.path.join(STATES_DIR, 'foraging_shivangi_no_sess1_clipped_session_index.json')) as f:
        index = json.load(f)
    return {s['session_id'].split('_')[1]: probs[s['start_idx']:s['end_idx'] + 1] for s in index}


def trial_table(session, states):
    """TrialIndex -> (state, reaction time), for the trials in Trial_Info (as erp_spectra_stats_RT.py)."""
    st = states[session]
    rt = np.load(os.path.join(STATES_DIR, 'processed', SESSION_FOLDERS[session], 'emissions.npy')).ravel()
    info = pd.read_pickle(os.path.join(TRIAL_INFO_DIR, session, 'Trial_Info.pkl'))
    info.iloc[:, 0] = (info.iloc[:, 0] - 1000).astype('Int64')
    df = pd.merge(info, pd.DataFrame({'TrialIndex': np.arange(len(st)), 'State': st, 'RT': rt[:len(st)]}),
                  left_on='Trial_Number', right_on='TrialIndex', how='inner')
    return {int(r.TrialIndex): (int(r.State), float(r.RT)) for r in df.itertuples()}


# ── Saccades (as saccade_triggered_average.py) ────────────────────────────────────────────
def detect_saccades(sacc, session, fs_eye):
    pred = np.nan_to_num(sacc[f'{session}__pred_orig']).astype(int)
    x, y, nan_mask = sacc[f'{session}__x_orig'], sacc[f'{session}__y_orig'], sacc[f'{session}__nan_mask']
    d = np.diff(pred)
    on, off = np.where(d == 1)[0] + 1, np.where(d == -1)[0] + 1
    if off.size and (on.size == 0 or off[0] < on[0]):
        off = off[1:]
    if on.size and (off.size == 0 or on[-1] > off[-1]):
        on = on[:-1]
    n = min(len(on), len(off))
    min_dur = int(round(MIN_SACC_DUR_MS / 1000 * fs_eye))
    keep = [(a, b) for a, b in zip(on[:n], off[:n])
            if b - a >= min_dur and not nan_mask[a:b].any() and np.hypot(x[b] - x[a], y[b] - y[a]) >= MICRO_AMP_DEG]
    return (np.array([k[0] for k in keep], int), np.array([k[1] for k in keep], int))


def samples_to_log_time(session, samples):
    folder = os.path.join(EYE_DIR, session)
    eye_file = next(os.path.basename(f).replace('.csv', '') for f in glob.glob(os.path.join(folder, '*.csv'))
                    if 'net.csv' not in os.path.basename(f))
    offset = tc.align_irec(os.path.join(folder, SESSION_LOGS[session]), os.path.join(folder, eye_file + 'net.csv'))
    pos_t = pd.read_csv(os.path.join(folder, eye_file + '.csv'), usecols=[0]).to_numpy().ravel()
    out = np.full(len(samples), np.nan)
    ok = samples < len(pos_t)
    out[ok] = pos_t[samples[ok]] + offset
    return out


def stimulus_onsets(session):
    with TextLog(os.path.join(EYE_DIR, session, SESSION_LOGS[session])) as log:
        evt, ts, _, _ = log.parse_eventmarkers()
    return ts[evt == 3011]


def session_saccades(sacc, session):
    on, off = detect_saccades(sacc, session, int(sacc['fs']))
    on_t, off_t = samples_to_log_time(session, on), samples_to_log_time(session, off)
    ok = np.isfinite(on_t) & np.isfinite(off_t)
    on_t, off_t = on_t[ok], off_t[ok]
    order = np.argsort(on_t)
    on_t, off_t = on_t[order], off_t[order]
    keep = np.ones(len(on_t), bool)
    keep[:-1] = np.diff(on_t) >= NEXT_SACC_GAP_MS / 1000
    return on_t[keep], off_t[keep]


# ── Per-session channel means ─────────────────────────────────────────────────────────────
def session_means(session, states, sacc):
    trials = trial_table(session, states)
    stim_t = stimulus_onsets(session)
    sac_on, sac_off = session_saccades(sacc, session)
    lfp = spy.load(os.path.join(LFP_DIR, session, 'Cleaned_lfp_FT.spy'))
    if lfp.trialdefinition.shape[1] < 4:
        lfp.trialdefinition = np.hstack((lfp.trialdefinition, np.arange(len(lfp.trialdefinition))[:, None]))
    fs = float(lfp.samplerate)
    ch_num = np.array([int(c.split('_')[-1]) for c in lfp.channel])       # 1-based channel numbers
    win = {e: (int(round(pre * fs)), int(round(post * fs))) for e, (pre, post) in EVENTS.items()}
    S = {e: np.zeros((len(STATES), sum(win[e]), N_CH)) for e in EVENTS}
    C = {e: np.zeros((len(STATES), N_CH)) for e in EVENTS}
    n_ep = {e: np.zeros(len(STATES), int) for e in EVENTS}

    for p, trl in enumerate(lfp.trialdefinition[:, 3].astype(int)):
        if trl not in trials or trials[trl][0] not in STATES or trl >= len(stim_t):
            continue
        k = STATES.index(trials[trl][0])
        x = np.asarray(lfp.trials[p], float)                               # samples x channels
        t = np.asarray(lfp.time[p], float)
        if np.all(np.isnan(x)):
            continue
        events = {'stimulus': [0.0], 'reaction': [trials[trl][1]] if np.isfinite(trials[trl][1]) else [],
                  'saccade_onset': sac_on - stim_t[trl], 'saccade_offset': sac_off - stim_t[trl]}
        for e, times in events.items():
            pre, post = win[e]
            for te in np.asarray(times, float):
                if te < t[0] or te > t[-1]:
                    continue
                i = int(np.searchsorted(t, te))
                if i - pre < 0 or i + post > len(t):
                    continue
                seg = x[i - pre:i + post]
                ok = ~np.isnan(seg).any(axis=0)
                if not ok.any():
                    continue
                S[e][k][:, ch_num[ok] - 1] += seg[:, ok]
                C[e][k][ch_num[ok] - 1] += 1
                n_ep[e][k] += 1
    out = {}
    for e in EVENTS:
        with np.errstate(invalid='ignore', divide='ignore'):
            out[e] = S[e] / np.where(C[e] > 0, C[e], np.nan)[:, None, :]
        print(f'  {e:15s} epochs: ' + ', '.join(f'state {s} {n_ep[e][k]}' for k, s in enumerate(STATES)))
    return out, n_ep, fs


# ── Session-label permutation test (as erp_spectra_stats.py) ──────────────────────────────
def permutation_test(d1, d2):
    pooled = np.vstack([d1, d2])
    labels = np.array([0] * len(d1) + [1] * len(d2))
    real = np.nanmean(d1, axis=0) - np.nanmean(d2, axis=0)
    mx, mn = np.zeros(N_PERMS), np.zeros(N_PERMS)
    for i in range(N_PERMS):
        rng.shuffle(labels)
        diff = np.nanmean(pooled[labels == 0], axis=0) - np.nanmean(pooled[labels == 1], axis=0)
        mx[i], mn[i] = np.nanmax(diff), np.nanmin(diff)
    lo, hi = np.percentile(mn, 100 * ALPHA / 2), np.percentile(mx, 100 * (1 - ALPHA / 2))
    return real, (real > hi) | (real < lo), (lo, hi)


def main():
    states = session_states()
    sacc = np.load(SACC_NPZ, allow_pickle=True)
    means, counts, fs = {}, {}, None
    for s in SESSIONS:
        print(f'\n=== {s} ===')
        means[s], counts[s], fs = session_means(s, states, sacc)

    res = {'states': np.array(STATES), 'sessions': np.array(SESSIONS), 'events': np.array(list(EVENTS))}
    print('\n=== state 0 vs state 2, per array ===')
    for e, (pre, post) in EVENTS.items():
        n = int(round(pre * fs)) + int(round(post * fs))
        x = (np.arange(n) - int(round(pre * fs))) / fs
        # session x state x array x time: mean over the array's channels with data
        tr = np.full((len(SESSIONS), len(STATES), N_ARRAYS, n), np.nan)
        for i, s in enumerate(SESSIONS):
            for a in range(N_ARRAYS):
                block = means[s][e][:, :, a * CH_PER_ARRAY:(a + 1) * CH_PER_ARRAY]
                has = np.isfinite(block).any(axis=1)                    # state x channel
                for k in range(len(STATES)):
                    if has[k].any():
                        tr[i, k, a] = np.nanmean(block[k][:, has[k]], axis=1)
        sig = np.zeros((N_ARRAYS, n), bool)
        thr = np.full((N_ARRAYS, 2), np.nan)
        n_sess = np.zeros(N_ARRAYS, int)
        for a in range(N_ARRAYS):
            d1 = tr[:, 0, a][np.isfinite(tr[:, 0, a]).all(axis=1)]
            d2 = tr[:, 1, a][np.isfinite(tr[:, 1, a]).all(axis=1)]
            n_sess[a] = min(len(d1), len(d2))
            if len(d1) < 2 or len(d2) < 2:
                continue
            diff, sig[a], thr[a] = permutation_test(d1, d2)
            spans = np.flatnonzero(np.diff(np.r_[0, sig[a].astype(int), 0]))
            print(f'  {e:15s} array {a + 1}: {int(sig[a].sum())}/{n} samples significant; spans (s): ' +
                  (', '.join(f'{x[spans[j]]:+.3f} to {x[spans[j + 1] - 1]:+.3f}' for j in range(0, len(spans), 2)) or '-') +
                  f'; max |diff| {np.nanmax(np.abs(diff)):.1f} uV, thr {thr[a, 0]:.1f}/{thr[a, 1]:.1f}')
        res.update({f'{e}_x': x, f'{e}_traces': tr, f'{e}_sig': sig, f'{e}_thr': thr, f'{e}_n_sessions': n_sess,
                    f'{e}_n_epochs': np.array([counts[s][e] for s in SESSIONS])})
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez_compressed(OUT, **res)
    print(f'\nwrote {OUT}')


if __name__ == '__main__':
    main()
