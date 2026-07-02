"""
=============================================================================================
Generate all regressors for the GLM design matrix.
=============================================================================================
Pipeline position: step 1 (see README.md). The first step: writes the per-channel regressor
.npz files (+ full_mask / full_mask_reg) that NeuralData.py (step 2) and DesignMatrix.py (step 3)
consume.

Sessions run in a for loop. The channel-independent work (binary/event/block regressors and
the masks) is computed once per session; the per-channel RF work (the expensive RF_stim h5
walk) is parallelised across SLURM workers with acme. Each channel is saved to its own folder.

BINARY:  correct, wrong, diff_easy, diff_hard, movement_left, movement_right,
         state_<k> (one per cognitive state, from erp_spectra_stats.py)
EVENT:   trial_onset, stim_onset, reward_onset, block_onset, reaction_onset,
         saccade_onset, saccade_offset (microsaccades excluded),
         {target, distractor, sky, mountain, grass}_in_RF_onset/offset

  target / distractor  <- RF_stim_collapse.h5 (leaves A/B)
  sky / mountain / grass <- RF_background.h5 (background bands from background_parsing.py)

Run in the warping env (needs acme: `conda install -c conda-forge esi-acme`).
"""

import os
import sys
import glob
import json
import pickle

import numpy as np
import pandas as pd
import syncopy as spy
import h5py

from acme import ParallelMap

sys.path.insert(1, '/cs/projects/MWzeronoise/Analysis/4Shivangi/code/functions/unreal_logfile')
sys.path.insert(1, '/cs/projects/MWzeronoise/Analysis/4Shivangi/code/functions/eyetracking')
from parse_logfile import TextLog
import time_conversion as tc   # iRec <-> log time alignment (for saccade timing)

# -------------------------
# Config
# -------------------------
data_dir = '/cs/projects/MWzeronoise/Analysis/4Shivangi/Datasets'
lfp_data_dir = os.path.join(data_dir, 'neural_data/stimAalign_cut/clean_full_length')
log_file_dir = os.path.join(data_dir, 'raw_data')
# make glm_config (single source of truth for the output tree + sampling rate) importable
for _d in (os.path.dirname(os.path.abspath(__file__)),
           os.path.dirname(os.path.dirname(os.path.abspath(__file__)))):
    if os.path.exists(os.path.join(_d, 'glm_config.py')):
        sys.path.insert(0, _d)
        break
from glm_config import RESULTS_DIR
results_dir = RESULTS_DIR
rf_base_dir = '/cs/projects/MWzeronoise/Analysis/4Shivangi/Results/RF_VR_mapping/RFarea_stim'
processed_dir = '/cs/projects/MWzeronoise/Analysis/4Shivangi/Datasets/states_analysis/processed' # reaction time (RT) per trial, in seconds

sessions = ['20230214']
channels = [2]

# SLURM / parallelisation config
SLURM_PARTITION = '8GBS'
MAX_WORKERS = 100
MEM_PER_WORKER = '8GB'

# per-trial cognitive state (one binary regressor per state, see erp_spectra_stats.py)
states_data_dir = '/cs/projects/MWzeronoise/Analysis/4Shivangi/Datasets/states_analysis'
N_STATES_TO_USE = 4

# saccades detection and timing
eye_data_dir = '/cs/projects/MWzeronoise/Analysis/4Shivangi/Datasets/eye_data'
sacc_npz_path = '/cs/projects/MWzeronoise/Analysis/4Shivangi/Results/saccade_detection/stitched_sessions.npz'
MIN_SACC_DUR_MS = 6.0   # microsaccade exclusion: shorter saccades are dropped
MICRO_AMP_DEG = 1.0     # microsaccade exclusion: smaller-amplitude saccades are dropped

# RF categories (sky/mountain/grass labels must match background_parsing.py)
RF_CATEGORIES = ['target', 'distractor', 'sky', 'mountain', 'grass']

# -------------------------
# Helpers
# -------------------------

def find_logfile(log_dir, session_name):
    """Find the single unreal .log for a session inside log_dir/session_name."""
    session_dir = os.path.join(log_dir, session_name)
    log_files = sorted(glob.glob(os.path.join(session_dir, '*.log')))
    if len(log_files) == 0:
        raise FileNotFoundError(f"No .log file found in {session_dir}")
    if len(log_files) > 1:
        raise ValueError(f"Multiple .log files found in {session_dir}:\n" + "\n".join(log_files))
    return log_files[0]


def remove_nan_trials_channels(datas):
    """Drop all-NaN trials/channels left in place when the data was cleaned."""
    trial_mask = [not np.all(np.isnan(tr)) for tr in datas.trials]
    if not any(trial_mask):
        return None, [], trial_mask
    trial_stack = np.concatenate(datas.trials, axis=0)
    valid_ch_idx = np.where(~np.all(np.isnan(trial_stack), axis=0))[0]
    if len(valid_ch_idx) == 0:
        return None, [], trial_mask
    datas_clean = spy.selectdata(spy.StructDict(channel=valid_ch_idx), datas)
    valid_channels = [datas.channel[i] for i in valid_ch_idx]
    return datas_clean, valid_channels, trial_mask


def ensure_trialindex_in_trialdefinition(datalfp):
    """Ensure trialdefinition has a 4th column holding the trial index (0..nTrials-1)."""
    try:
        if datalfp.trialdefinition.shape[1] < 4:
            nTrials = datalfp.trialdefinition.shape[0]
            datalfp.trialdefinition = np.hstack(
                (datalfp.trialdefinition, np.arange(nTrials).reshape(-1, 1)))
    except Exception as e:
        print("Warning: could not ensure trialindex in trialdefinition:", e)


def detect_saccades(sacc_npz, session, fs_eye):
    """(onset, offset) eye-sample pairs of non-micro saccades (microsaccades excluded).
    Mirrors saccade_erp_rf_split.py: drops saccades shorter than MIN_SACC_DUR_MS, any that
    overlap a NaN gap, or with amplitude < MICRO_AMP_DEG."""
    pred = np.nan_to_num(sacc_npz[f'{session}__pred_orig']).astype(int)
    x = sacc_npz[f'{session}__x_orig']
    y = sacc_npz[f'{session}__y_orig']
    nan_mask = sacc_npz[f'{session}__nan_mask']
    min_dur_samp = int(round(MIN_SACC_DUR_MS / 1000 * fs_eye))

    d = np.diff(pred)
    onsets = np.where(d == 1)[0] + 1
    offsets = np.where(d == -1)[0] + 1
    if offsets.size and (onsets.size == 0 or offsets[0] < onsets[0]):
        offsets = offsets[1:]
    if onsets.size and (offsets.size == 0 or onsets[-1] > offsets[-1]):
        onsets = onsets[:-1]
    n = min(len(onsets), len(offsets))
    onsets, offsets = onsets[:n], offsets[:n]

    keep_on, keep_off = [], []
    for on, off in zip(onsets, offsets):
        if off - on < min_dur_samp:                                  # too short -> microsaccade
            continue
        if nan_mask[on:off].any():                                   # overlaps a gap
            continue
        if np.hypot(x[off] - x[on], y[off] - y[on]) < MICRO_AMP_DEG:  # too small -> microsaccade
            continue
        keep_on.append(on)
        keep_off.append(off)
    return np.asarray(keep_on, dtype=int), np.asarray(keep_off, dtype=int)


def samples_to_log_time(session, samples):
    """Convert eye-sample indices (iRec rows) -> log/Unreal time (s); NaN beyond the position file."""
    folder = os.path.join(eye_data_dir, session)
    log_path = sorted(glob.glob(os.path.join(folder, '*.log')))[0]
    eye_file = next(os.path.basename(f).replace('.csv', '')
                    for f in glob.glob(os.path.join(folder, '*.csv'))
                    if 'net.csv' not in os.path.basename(f))
    log_irec_offset = tc.align_irec(log_path, os.path.join(folder, eye_file + 'net.csv'))
    pos_t = pd.read_csv(os.path.join(folder, eye_file + '.csv'), usecols=[0]).to_numpy().ravel()
    valid = samples < len(pos_t)
    out = np.full(len(samples), np.nan)
    out[valid] = pos_t[samples[valid]] + log_irec_offset
    return out


# -------------------------
# RF helpers (geometry/logic unchanged)
# -------------------------

def find_point_name(channel, rf_stim_h5):
    """Return the 'Point_{channel}_...' group name used in the RF h5 for this channel."""
    prefix = f'Point_{channel}_'
    with h5py.File(rf_stim_h5, 'r') as f:
        tname = sorted(f.keys(), key=lambda n: int(n.split('_')[-1]))[0]
        tg = f[tname]
        tpname = sorted(tg.keys(), key=lambda n: int(n.split('_')[-1]))[0]
        for pname in tg[tpname].keys():
            if pname.startswith(prefix):
                return pname
    raise ValueError(f"No point named '{prefix}*' found in {rf_stim_h5}")


def extract_rf_lists(point_name, rf_stim_h5, rf_background_h5, target_stim_per_trial, correct):
    """
    Walk the per-frame RF h5 files for one RF point and return, per trial, the time-point
    sequences of where that RF sits:
        target / distractor    (RF_stim_collapse.h5, leaves A/B)
        sky / mountain / grass  (RF_background.h5, background bands from background_parsing.py)
    Each list holds one array per trial, length = number of time points in that trial.
    """
    lists = {k: [] for k in RF_CATEGORIES}

    with h5py.File(rf_stim_h5, 'r') as sf, h5py.File(rf_background_h5, 'r') as bf:
        trial_names = sorted(sf.keys(), key=lambda n: int(n.split('_')[-1]))

        for trial_num, tname in enumerate(trial_names):
            stim_tg = sf[tname]
            bg_tg = bf[tname] if tname in bf else None

            timepoint_names = sorted(stim_tg.keys(), key=lambda n: int(n.split('_')[-1]))
            n_tp = len(timepoint_names)

            tr = {k: np.zeros(n_tp) for k in RF_CATEGORIES}
            target_stim = target_stim_per_trial[trial_num]   # 'A' or 'B'

            for i, tpname in enumerate(timepoint_names):
                stim_tpg = stim_tg[tpname]

                # Is this a collapse frame?
                collapsed_case = bool(stim_tpg.attrs.get('collapsed_case', False))

                point_group = stim_tpg[point_name]
                in_A = bool(point_group['inside_transformed_outline_A'][()])
                in_B = bool(point_group['inside_transformed_outline_B'][()])

                # --- target / distractor (same logic as the original RF extraction) ---
                # In collapsed case both stimuli are in RF, we keep only the one that is reached
                if collapsed_case:
                    if correct[trial_num] == 1 and target_stim == 'A':
                        if in_A:
                            tr['target'][i] = 1
                    elif correct[trial_num] == 0 and target_stim == 'A':
                        if in_B:
                            tr['distractor'][i] = 1
                    elif correct[trial_num] == 1 and target_stim == 'B':
                        if in_B:
                            tr['target'][i] = 1
                    elif correct[trial_num] == 0 and target_stim == 'B':
                        if in_A:
                            tr['distractor'][i] = 1
                # If it is not in collapsed case, find target identity and store accordingly
                else:
                    if in_A:
                        if target_stim == 'A':
                            tr['target'][i] = 1
                        else:
                            tr['distractor'][i] = 1
                    elif in_B:
                        if target_stim == 'B':
                            tr['target'][i] = 1
                        else:
                            tr['distractor'][i] = 1

                # --- sky / mountain / grass (background bands from background_parsing.py) ---
                # RF_background.h5 only contains BACKGROUND points (RF outside every stimulus),
                # so a missing point here means the RF was on a stimulus at this time point.
                if bg_tg is not None and tpname in bg_tg and point_name in bg_tg[tpname]:
                    label = bg_tg[tpname][point_name]['background'][()]
                    label = label.decode() if isinstance(label, (bytes, bytearray)) else str(label)
                    if label in tr:
                        tr[label][i] = 1   # 'unknown' is ignored (stays all-zero)

            for k in RF_CATEGORIES:
                lists[k].append(tr[k])

    return lists


def load_or_extract_rf_lists(point_name, output_path, rf_stim_h5, rf_background_h5,
                             target_stim_per_trial, correct):
    """Load the cached per-trial RF lists from output_path, or extract and cache them."""
    paths = {k: os.path.join(output_path, f'{k}_in_RF_list.npy') for k in RF_CATEGORIES}
    if all(os.path.exists(p) for p in paths.values()):
        return {k: np.load(paths[k], allow_pickle=True) for k in RF_CATEGORIES}

    print(f"Extracting RF lists for point {point_name} (this can take a while) ...")
    lists = extract_rf_lists(point_name, rf_stim_h5, rf_background_h5, target_stim_per_trial, correct)
    for k in RF_CATEGORIES:
        np.save(paths[k], np.array(lists[k], dtype=object))
    return lists


# -------------------------
# Session-level work (channel-independent, computed once per session)
# -------------------------

def compute_session(session_name):
    """Build all channel-independent regressors + masks for a session; returns a dict."""
    filename = find_logfile(log_file_dir, session_name)

    datalfp = spy.load(os.path.join(lfp_data_dir, session_name, 'Cleaned_lfp_FT.spy'))
    ensure_trialindex_in_trialdefinition(datalfp)
    data = spy.selectdata(spy.StructDict(), datalfp)
    datas_clean, _, trial_mask = remove_nan_trials_channels(data)

    with TextLog(filename) as log:
        df = pd.DataFrame(log.get_info_per_trial(return_eventmarkers=True, return_loc=False))
        evt, ts, _, _ = log.parse_eventmarkers()

    # full mask: drop NaN trials and no-response trials
    outcome_mask = (df["Correct"] != 0) | (df["Wrong"] != 0)
    full_mask = trial_mask & outcome_mask

    # binary regressors (invalid trials set to 0)
    correct = df['Correct'].values * full_mask
    wrong = df['Wrong'].values * full_mask
    diff_hard = ((df['MorphTarget'] == 51) | (df['MorphTarget'] == 49)).astype(int).values * full_mask
    diff_easy = ((df['MorphTarget'] == 70) | (df['MorphTarget'] == 30)).astype(int).values * full_mask
    movement_left = (((df['Left'] == 1) & (df['Correct'] == 1)) |
                     ((df['Right'] == 1) & (df['Wrong'] == 1))).astype(int).values * full_mask
    movement_right = (((df['Right'] == 1) & (df['Correct'] == 1)) |
                      ((df['Left'] == 1) & (df['Wrong'] == 1))).astype(int).values * full_mask

    # per-trial state -> one binary regressor per state (states from erp_spectra_stats.py)
    state_probs = np.load(os.path.join(states_data_dir, 'foraging_shivangi_no_sess1_clipped_state_assignments.npy'))
    with open(os.path.join(states_data_dir, 'foraging_shivangi_no_sess1_clipped_session_index.json')) as f:
        session_index = json.load(f)
    session_to_probs = {sess['session_id'].split('_')[1]: state_probs[sess['start_idx']: sess['end_idx'] + 1]
                        for sess in session_index}
    predicted_states = session_to_probs.get(session_name, np.array([]))

    n_trials = len(full_mask)
    states_per_trial = np.full(n_trials, -1, dtype=int)
    m = min(n_trials, len(predicted_states))
    states_per_trial[:m] = np.asarray(predicted_states[:m]).astype(int)

    # the four states = first N_STATES_TO_USE sorted unique state values (as in erp_spectra_stats.py)
    unique_states = np.sort(np.unique(predicted_states))[:N_STATES_TO_USE] if predicted_states.size else []
    state_regressors = {f'state_{int(s)}': (states_per_trial == int(s)).astype(int) * full_mask
                        for s in unique_states}

    # event onset times from eventmarkers
    trial_onset_ts = ts[np.where(evt == 3000)[0]]
    trial_end_ts = ts[np.where(evt == 3090)[0]]
    stim_onset_ts = ts[np.where(evt == 3011)[0]]
    reward_onset_ts = ts[np.where((evt >= 5000) & (evt < 6000))[0]]

    trial_onset_ts = trial_onset_ts[:-1]   # drop the last (non-existent) trial
    stim_onset_ts = stim_onset_ts[:-1]

    dur_stim = np.asarray(stim_onset_ts - trial_onset_ts, dtype=float)
    dur_reward = []
    for i in range(len(trial_onset_ts)):
        r = reward_onset_ts[(reward_onset_ts > trial_onset_ts[i]) & (reward_onset_ts < trial_end_ts[i])]
        dur_reward.append(r[0] - trial_onset_ts[i] if len(r) > 0 else 0)
    dur_reward = np.asarray(dur_reward, dtype=float)

    fs = datas_clean.samplerate
    dur_stim_idx = np.floor(dur_stim * fs).astype(int)
    dur_reward_idx = np.floor(dur_reward * fs).astype(int)

    # per-trial onset regressors (1 at the event sample); events outside the trial's LFP
    # window are dropped (not clamped), same convention as the reaction/saccade onsets below
    trial_onset_regs, stim_onset_regs, reward_onset_regs = [], [], []
    for i, trial in enumerate(datas_clean.trials):
        n = len(trial)
        tt = np.zeros(n); tt[0] = 1
        ss = np.zeros(n)
        if 0 <= dur_stim_idx[i] < n:
            ss[dur_stim_idx[i]] = 1
        rr = np.zeros(n)
        if 0 < dur_reward_idx[i] < n:   # dur_reward_idx == 0 marks no-reward trials
            rr[dur_reward_idx[i]] = 1
        trial_onset_regs.append(tt)
        stim_onset_regs.append(ss)
        reward_onset_regs.append(rr)

    stim_onset_raw = np.concatenate(stim_onset_regs)
    trial_onset_raw = np.concatenate(trial_onset_regs)
    trial_onset = np.concatenate([reg * full_mask[i] for i, reg in enumerate(trial_onset_regs)])
    stim_onset = np.concatenate([reg * full_mask[i] for i, reg in enumerate(stim_onset_regs)])
    reward_onset = np.concatenate([reg * full_mask[i] for i, reg in enumerate(reward_onset_regs)])

    # reaction-time event regressor: 1 at the RT sample within each trial.
    # RT (emissions.npy) is per trial in seconds, measured from STIM onset; trial onset is
    # dur_stim earlier, so the sample is floor((dur_stim + rt)*fs) -- same trial-onset-relative
    # convention as the stim/reward/saccade onsets above.
    emissions_matches = glob.glob(os.path.join(processed_dir, f'Cosmos_{session_name}_*', 'emissions.npy'))
    rt_values = np.load(emissions_matches[0]).flatten() if emissions_matches else np.array([])
    if rt_values.size == 0:
        print(f"[{session_name}] WARNING: emissions.npy not found -> reaction_onset is all zeros")
    reaction_onset_regs = []
    for i, trial in enumerate(datas_clean.trials):
        n = len(trial)
        rr = np.zeros(n)
        tidx = int(datas_clean.trialdefinition[i, 3])   # trial index (matches the RT array)
        if tidx < len(rt_values) and np.isfinite(rt_values[tidx]):
            samp = int(np.floor((dur_stim[i] + rt_values[tidx]) * fs))
            if 0 <= samp < n:
                rr[samp] = 1
        reaction_onset_regs.append(rr)
    reaction_onset_raw = np.concatenate(reaction_onset_regs)
    reaction_onset = np.concatenate([reg * full_mask[i] for i, reg in enumerate(reaction_onset_regs)])

    # saccade onset/offset event regressors (microsaccades excluded).
    # Saccade times (eye samples) -> log time, then per trial relative to stim onset == LFP
    # trial time, mapped to the nearest LFP sample (see saccade_erp_rf_split.py).
    on_log = off_log = np.array([])
    try:
        sacc_npz = np.load(sacc_npz_path, allow_pickle=True)
        if f'{session_name}__pred_orig' in sacc_npz:
            onsets, offsets = detect_saccades(sacc_npz, session_name, int(sacc_npz['fs']))
            on_log = samples_to_log_time(session_name, onsets)
            off_log = samples_to_log_time(session_name, offsets)
            good = ~(np.isnan(on_log) | np.isnan(off_log))
            on_log, off_log = on_log[good], off_log[good]
        else:
            print(f"[{session_name}] WARNING: no saccade data -> saccade regressors are all zeros")
    except Exception as e:
        print(f"[{session_name}] WARNING: saccade extraction failed ({e}) -> saccade regressors are all zeros")

    # placed at floor((t - trial_onset)*fs), the same convention as the stim/reward onsets
    # above; only the per-trial sample count comes from the LFP, not its time axis
    saccade_onset_regs, saccade_offset_regs = [], []
    for i, trial in enumerate(datas_clean.trials):
        n = len(trial)
        son, soff = np.zeros(n), np.zeros(n)
        if on_log.size:
            on_smp = np.floor((on_log - trial_onset_ts[i]) * fs).astype(int)
            off_smp = np.floor((off_log - trial_onset_ts[i]) * fs).astype(int)
            in_trl = (on_smp >= 0) & (on_smp < n)          # saccade onset inside this trial
            son[on_smp[in_trl]] = 1
            soff[np.clip(off_smp[in_trl], 0, n - 1)] = 1
        saccade_onset_regs.append(son)
        saccade_offset_regs.append(soff)
    saccade_onset_raw = np.concatenate(saccade_onset_regs)
    saccade_offset_raw = np.concatenate(saccade_offset_regs)
    saccade_onset = np.concatenate([reg * full_mask[i] for i, reg in enumerate(saccade_onset_regs)])
    saccade_offset = np.concatenate([reg * full_mask[i] for i, reg in enumerate(saccade_offset_regs)])

    # block onset: 1 at the first sample of each block
    block_exit_ts = np.append(ts[np.where(evt == 3091)[0]], trial_end_ts[-1])
    block_end_trials = [i for i in range(len(trial_end_ts))
                        if np.any(np.isclose(block_exit_ts, trial_end_ts[i]))]
    block_onset_list, switch_block = [], True
    for i, trial in enumerate(datas_clean.trials):
        vals = np.zeros(len(trial))
        if switch_block:
            vals[0] = 1
            switch_block = False
        block_onset_list.append(vals)
        if i in block_end_trials:
            switch_block = True
    block_onset = np.concatenate(block_onset_list)

    # mask at regressor (sample) length
    full_mask_reg = np.concatenate(
        [np.full(len(trial), full_mask[i], dtype=bool) for i, trial in enumerate(datas_clean.trials)])

    target_stim_per_trial = np.where(df['Right'].values == 1, 'A', 'B')

    return dict(
        full_mask=full_mask, full_mask_reg=full_mask_reg,
        correct=correct, wrong=wrong, diff_hard=diff_hard, diff_easy=diff_easy,
        movement_left=movement_left, movement_right=movement_right,
        state_regressors=state_regressors,
        trial_onset=trial_onset, stim_onset=stim_onset, reward_onset=reward_onset,
        block_onset=block_onset, reaction_onset=reaction_onset,
        saccade_onset=saccade_onset, saccade_offset=saccade_offset,
        trial_onset_raw=trial_onset_raw, stim_onset_raw=stim_onset_raw,
        reaction_onset_raw=reaction_onset_raw,
        saccade_onset_raw=saccade_onset_raw, saccade_offset_raw=saccade_offset_raw,
        target_stim_per_trial=target_stim_per_trial)


# -------------------------
# Per-channel worker: RF regressors + save
# -------------------------

def process_channel(channel, session_name, bundle_path, rf_base_dir, results_dir):
    """Build the RF regressors for one channel from the session bundle and save everything."""
    with open(bundle_path, 'rb') as f:
        b = pickle.load(f)

    rf_h5_dir = os.path.join(rf_base_dir, session_name)
    rf_stim_h5 = os.path.join(rf_h5_dir, 'RF_stim_collapse.h5')
    rf_background_h5 = os.path.join(rf_h5_dir, 'RF_background.h5')

    output_path = os.path.join(results_dir, session_name, f'channel{channel}_regressors')
    os.makedirs(output_path, exist_ok=True)

    # this channel's RF point and its per-trial RF time courses
    point_name = find_point_name(channel, rf_stim_h5)
    print(f"channel {channel} -> RF point {point_name}")
    rf_lists = load_or_extract_rf_lists(point_name, output_path, rf_stim_h5, rf_background_h5,
                                        b['target_stim_per_trial'], b['correct'])

    # map each per-trial list onto the full-session stimulus-onset timeline
    stim_onset_raw, stim_onset, full_mask_reg = b['stim_onset_raw'], b['stim_onset'], b['full_mask_reg']
    stim_onset_idx = np.where(stim_onset_raw == 1)[0]

    def place_on_timeline(per_trial_list):
        reg = np.zeros_like(stim_onset, dtype=float)
        for i, trial_reg in enumerate(per_trial_list):
            reg[stim_onset_idx[i]:stim_onset_idx[i] + len(trial_reg)] = trial_reg
        return reg * full_mask_reg   # set invalid trials to 0

    rf_regs = {cat: place_on_timeline(rf_lists[cat]) for cat in RF_CATEGORIES}

    rf_sum = sum(rf_regs.values())   # the five categories are mutually exclusive
    print(f"channel {channel}: RF length {len(rf_sum)}, any overlap (should be False): {np.any(rf_sum > 1)}")

    # onset / offset event regressors for each RF category, dropped to valid samples
    def onset_offset(reg):
        onset = np.zeros_like(reg); onset[1:] = (reg[:-1] == 0) & (reg[1:] == 1)
        offset = np.zeros_like(reg); offset[1:] = (reg[:-1] == 1) & (reg[1:] == 0)
        return onset.astype(int), offset.astype(int)

    rf_events = {}
    for cat, reg in rf_regs.items():
        onset, offset = onset_offset(reg)
        rf_events[f'{cat}_in_RF_onset'] = onset[full_mask_reg]
        rf_events[f'{cat}_in_RF_offset'] = offset[full_mask_reg]

    # drop invalid trials: binary regressors by trial, event regressors by sample
    full_mask = b['full_mask']
    binary = {k: b[k][full_mask] for k in
              ('correct', 'wrong', 'diff_hard', 'diff_easy', 'movement_left', 'movement_right')}
    events = {k: b[k][full_mask_reg] for k in
              ('trial_onset', 'stim_onset', 'reward_onset', 'block_onset', 'reaction_onset',
               'saccade_onset', 'saccade_offset')}

    # save everything
    for name, arr in binary.items():
        np.savez_compressed(os.path.join(output_path, f'{name}.npz'), data=arr)
    for name, arr in b['state_regressors'].items():   # one binary regressor per state
        np.savez_compressed(os.path.join(output_path, f'{name}.npz'), data=arr[full_mask])
    np.savez_compressed(os.path.join(output_path, 'trial_onset_raw.npz'), data=b['trial_onset_raw'])
    np.savez_compressed(os.path.join(output_path, 'reaction_onset_raw.npz'), data=b['reaction_onset_raw'])
    np.savez_compressed(os.path.join(output_path, 'saccade_onset_raw.npz'), data=b['saccade_onset_raw'])
    np.savez_compressed(os.path.join(output_path, 'saccade_offset_raw.npz'), data=b['saccade_offset_raw'])
    for name, arr in events.items():
        np.savez_compressed(os.path.join(output_path, f'{name}.npz'), data=arr)
    for name, arr in rf_events.items():
        np.savez_compressed(os.path.join(output_path, f'{name}.npz'), data=arr)
    np.savez_compressed(os.path.join(output_path, 'full_mask.npz'), data=full_mask)
    np.savez_compressed(os.path.join(output_path, 'full_mask_reg.npz'), data=full_mask_reg)

    print(f"channel {channel}: saved -> {output_path}")
    return output_path


# -------------------------
# Driver: sessions looped, channels parallelised
# -------------------------
if __name__ == '__main__':
    for session_name in sessions:
        print(f"\n[{session_name}] computing session-level regressors ...")
        bundle = compute_session(session_name)

        session_out = os.path.join(results_dir, session_name)
        os.makedirs(session_out, exist_ok=True)
        bundle_path = os.path.join(session_out, '_session_bundle.pkl')
        with open(bundle_path, 'wb') as f:
            pickle.dump(bundle, f)

        n_workers = min(MAX_WORKERS, len(channels))
        print(f"[{session_name}] {len(channels)} channels -> {n_workers} workers on '{SLURM_PARTITION}'")

        with ParallelMap(process_channel, channels, session_name, bundle_path, rf_base_dir, results_dir,
                         n_inputs=len(channels),
                         partition=SLURM_PARTITION,
                         n_workers=n_workers,
                         mem_per_worker=MEM_PER_WORKER,
                         write_worker_results=False,   # workers save their own npz; just return the path
                         setup_interactive=False) as pmap:
            pmap.compute()

        os.remove(bundle_path)
