"""
States vs behavior: relate HMM behavioral states to what the animal is doing.

Takes a trained 4-state HMM's per-trial labels (23 sessions) and crosses them
against trial outcome, difficulty, reaction time, and position within/around
experimental blocks -- testing each relationship and plotting it.

INPUTS
    state_assignments.npy : hard labels, argmax state per trial, shape (n_trials,).
    state_probs.npy       : soft labels, HMM posterior P(state=k | all trials),
                            shape (n_trials, K), rows sum to 1. argmax = hard label.
    session_index.json    : maps the flat trial index back to each session.
    emissions.npy + .log  : reaction time and eventmarkers (outcome, block change).

HARD vs SOFT
    Hard (counts / discrete identity): state durations, transition matrix, the
        ASR significance heatmaps, and grouping trials for RT.
    Soft (averaging membership): the baseline-ratio magnitude panels -- graded
        membership is far less noisy than a 0/1 label, especially for the rare
        states (1 and 3, ~1%).
    Rule of thumb: counts -> hard, averages -> soft.

CORE STATISTIC (the z-score heatmaps): ASR + circular-shift permutation
    For each (state, category) cell, build the 2x2 table [in-state vs not] x
    [in-category vs not] and compute the adjusted standardized residual:

        E   = row_total * col_total / N
        ASR = (O - E) / sqrt( E * (1 - p_row) * (1 - p_col) )    ,  p = total/N

    ASR ~ N(0,1) under independence, so it reads like a z-score: positive = the
    state occurs with that category MORE than chance, negative = less, magnitude
    = SDs from expected. The (1-p_row)(1-p_col) term makes it unit-variance
    (hence "adjusted") so cells are comparable. Computed on HARD labels (counts).

    Significance is NOT from free reshuffling -- states are sticky (highly
    autocorrelated), so free shuffling would treat trials as independent and
    over-call significance. Instead each of 10,000 permutations CIRCULARLY SHIFTS
    the state sequence within each session (np.roll): this preserves run-length
    structure and per-state counts, randomizing only the phase relative to the
    behavioral labels -- the correct null. Per cell:

        p = count(|ASR_perm| >= |ASR_obs|) / 10000      (two-tailed); * if p<0.05.

    asr_with_perm         -> trial-level tables (outcome, outcome x difficulty).
    asr_with_perm_records -> block-position / block-transition (shifts the full
                             session sequence, re-reads each record by abs index).

BASELINE-RATIO PANELS (magnitude, not significance): SOFT
    ratio = mean P(state | trial) over trials in a bin/offset
            -------------------------------------------------- ,  centered at 1.0
                  overall mean P(state)  (= baseline_soft)
    1.0 = as common as its overall rate, 2.0 = twice as common.

OTHER PANELS
    State durations : run-lengths of consecutive identical hard labels.
    Transition matrix: empirical count(a -> b), row-normalized.
    RT by state     : violin + pairwise Mann-Whitney U (Bonferroni) -- continuous
                      RT distributions per state, a different question than ASR.

ENV: run with the `warping` conda env (syncopy).
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd
from datetime import datetime
import seaborn as sns
from itertools import groupby, combinations
from scipy.stats import zscore, mannwhitneyu, wilcoxon
from scipy.stats import chi2_contingency, fisher_exact
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import json
import math

# --------------------------------------------------------------------
# Custom module paths
# --------------------------------------------------------------------
sys.path.insert(1, '/mnt/cs/projects/MWzeronoise/Analysis/4Shivangi/code/functions/unreal_logfile')
from parse_logfile import TextLog
from preprocessing import align_ephys as align
from preprocessing import snippet_ephys as snip
import syncopy as spy

# --------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------
states_data_dir = '/cs/projects/MWzeronoise/Analysis/4Shivangi/Datasets/states_analysis'
processed_dir = '/cs/projects/MWzeronoise/Analysis/4Shivangi/Datasets/states_analysis/processed'
output_dir = '/cs/projects/MWzeronoise/Analysis/4Shivangi/plots/states_analysis'
raw_data_dir = '/cs/projects/MWzeronoise/Analysis/4Shivangi/Datasets/raw_data'
os.makedirs(output_dir, exist_ok=True)

# --------------------------------------------------------------------
# Statistics helpers: adjusted standardized residuals (ASR)
#                     + circular-shift permutation null
# --------------------------------------------------------------------
# Why this approach: HMM states are "sticky" (a state persists for many
# consecutive trials), so trial-by-trial state labels are strongly
# autocorrelated. A plain reshuffle would treat trials as independent and
# inflate significance. A *circular shift* per session randomizes only the
# phase of the state sequence relative to the behavioral labels while keeping
# each session's run-length structure (and each state's count) intact -- the
# correct null for autocorrelated sequences.
def asr_matrix(state_arr, states, group_arrays):
    """Signed adjusted standardized residual for each (state, group) cell.

    For each state `st` and each binary behavioral mask `grp`, build the 2x2
    table [in-state vs not] x [in-group vs not] and return the ASR of the
    (in-state & in-group) cell:

        ASR = (O - E) / sqrt(E * (1 - p_row) * (1 - p_col))

    Returns an array of shape (n_states, n_groups).
    """
    state_arr = np.asarray(state_arr)
    N = len(state_arr)
    Z = np.full((len(states), len(group_arrays)), np.nan)
    for i, st in enumerate(states):
        in_state = (state_arr == st)
        col = in_state.sum()                      # invariant under circular shift
        p_col = col / N
        for j, grp in enumerate(group_arrays):
            row = grp.sum()
            if row == 0 or col == 0:
                continue
            O = np.count_nonzero(in_state & grp)  # only this changes per shift
            E = row * col / N
            p_row = row / N
            denom = math.sqrt(E * (1 - p_row) * (1 - p_col))
            if denom > 0:
                Z[i, j] = (O - E) / denom
    return Z


def _roll_within_sessions(state_flat, sess_slices, rng):
    """Circularly shift the state sequence within each session by a random offset."""
    shifted = state_flat.copy()
    for idx in sess_slices:
        n = len(idx)
        if n > 1:
            shifted[idx] = np.roll(state_flat[idx], rng.integers(1, n))
    return shifted


def asr_with_perm(state_flat, sess_slices, states, group_arrays, n_perms, rng):
    """Observed ASR matrix + circular-shift permutation p-values.

    `state_flat` is one ordered state value per row; `sess_slices` are the
    per-session contiguous row indices; `group_arrays` are FIXED behavioral
    masks aligned to those rows. Each permutation circularly shifts the states
    within session, recomputes the ASR matrix, and counts how often the
    permuted |ASR| >= observed |ASR| (two-tailed). Returns (asr, p), each
    shape (n_states, n_groups).
    """
    obs = asr_matrix(state_flat, states, group_arrays)
    valid = ~np.isnan(obs)
    count = np.zeros_like(obs)
    for _ in range(n_perms):
        perm = asr_matrix(_roll_within_sessions(state_flat, sess_slices, rng),
                          states, group_arrays)
        count[valid] += (np.abs(perm) >= np.abs(obs))[valid]
    p = np.full_like(obs, np.nan)
    p[valid] = count[valid] / n_perms
    return obs, p


def asr_with_perm_records(rec_session, rec_t, group_arrays, states,
                          state_seqs, n_perms, rng):
    """ASR + circular-shift null for record-based analyses (block position /
    transitions), whose rows are (session, absolute-trial) records drawn from
    the full session sequences.

    Each permutation circularly shifts each FULL session sequence and re-reads
    the state for every record at its absolute trial index. This keeps the
    shift defined on the true trial order even when records are a subset or
    repeat trials across overlapping windows. Returns (asr, p),
    shape (n_states, n_groups).
    """
    rec_session = np.asarray(rec_session)
    rec_t = np.asarray(rec_t)
    sess_rows = {s: np.where(rec_session == s)[0] for s in np.unique(rec_session)}
    sess_abst = {s: rec_t[rows] for s, rows in sess_rows.items()}

    def gather(seqs):
        out = np.empty(len(rec_session), dtype=int)
        for s, rows in sess_rows.items():
            out[rows] = seqs[s][sess_abst[s]]
        return out

    obs = asr_matrix(gather(state_seqs), states, group_arrays)
    valid = ~np.isnan(obs)
    count = np.zeros_like(obs)
    for _ in range(n_perms):
        shifted = {s: (np.roll(seq, rng.integers(1, len(seq))) if len(seq) > 1 else seq)
                   for s, seq in state_seqs.items()}
        perm = asr_matrix(gather(shifted), states, group_arrays)
        count[valid] += (np.abs(perm) >= np.abs(obs))[valid]
    p = np.full_like(obs, np.nan)
    p[valid] = count[valid] / n_perms
    return obs, p

# --------------------------------------------------------------------
# User Config
# --------------------------------------------------------------------
N_STATES_TO_USE = 4
states_to_include = list(range(N_STATES_TO_USE))

# --------------------------------------------------------------------
# Load states info for all sessions (from centralized state assignments)
# --------------------------------------------------------------------
print("[1/9] Loading state assignments and session index...")
# Hard vs soft state labels:
#   state_assignments  -> hard labels: the argmax state per trial, shape (n_trials,).
#                         Used for everything that needs a discrete state identity:
#                         durations, the transition matrix, the count-based ASR
#                         heatmaps (outcome / difficulty / block-position-ASR /
#                         block-transition-ASR) and grouping trials for RT.
#   state_probs        -> soft labels: the HMM forward-backward posterior
#                         P(state = k | all trials) per trial, shape (n_trials, K),
#                         each row summing to 1. The hard label is just its argmax.
#                         Soft retains graded membership, so it is far less noisy
#                         when AVERAGING membership across trials -- especially for
#                         the rare states (1 and 3, ~1% each), where a hard 0/1 is
#                         almost pure noise. We therefore use soft probabilities for
#                         the baseline-ratio (magnitude) panels below.
state_assignments = np.load(f'{states_data_dir}/foraging_shivangi_no_sess1_clipped_state_assignments.npy')
state_probs = np.load(f'{states_data_dir}/foraging_shivangi_no_sess1_clipped_state_probs.npy')
with open(f'{states_data_dir}/foraging_shivangi_no_sess1_clipped_session_index.json') as f:
    session_index = json.load(f)

# Derive sessions dynamically from session_index and processed directory
session_folders = {}
for sess in session_index:
    sid = sess['session_id']
    date = sid.split('_')[1]
    proc_path = os.path.join(processed_dir, sid)
    raw_path = os.path.join(raw_data_dir, date)
    if os.path.isdir(proc_path) and os.path.isdir(raw_path):
        session_folders[date] = sid
sessions = sorted(session_folders.keys())

state_colors = {
    0: (0.55, 0.0, 0.55),   # purple
    1: (0.0, 0.39, 0.39),   # teal
    2: (0.8, 0.33, 0.0),    # orange
    3: (0.25, 0.35, 0.55)   # slate blue
}

session_to_states = {}   # hard labels per session, shape (n_trials,)
session_to_probs = {}    # soft posteriors per session, shape (n_trials, K)
for sess in session_index:
    session_id = sess['session_id']
    session_date = session_id.split('_')[1]
    session_to_states[session_date] = state_assignments[
        sess['start_idx']: sess['end_idx'] + 1
    ]
    session_to_probs[session_date] = state_probs[
        sess['start_idx']: sess['end_idx'] + 1
    ]

# --------------------------------------------------------------------
# Load RT (emissions) for each session
# --------------------------------------------------------------------
print(f"[2/9] Loading RT emissions for {len(sessions)} sessions...")
session_to_rt = {}
for session_name in sessions:
    emissions_path = os.path.join(processed_dir, session_folders[session_name], 'emissions.npy')
    if os.path.exists(emissions_path):
        session_to_rt[session_name] = np.load(emissions_path).flatten()
    else:
        print(f"  emissions.npy not found for {session_name}")
        session_to_rt[session_name] = None

# --------------------------------------------------------------------
# State durations (pooled across sessions)
# --------------------------------------------------------------------
print("[3/9] Computing state durations and saving plot...")
all_states = []
for session_name in sessions:
    if session_name not in session_to_states:
        continue
    predicted_states = session_to_states[session_name]
    all_states.extend(predicted_states.ravel().tolist())

state_lengths, state_ids = [], []
for state, group in groupby(all_states):
    state_lengths.append(len(list(group)))
    state_ids.append(state)

df_state_lengths = pd.DataFrame({'state': state_ids, 'length': state_lengths})
plt.figure(figsize=(8, 4))
sns.histplot(data=df_state_lengths, x='length', hue='state', multiple='stack', bins=30)
plt.title("State Durations Across All Sessions")
plt.xlabel("State Duration (trials)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'state_duration.pdf'))
plt.close()

# --------------------------------------------------------------------
# State transitions
# --------------------------------------------------------------------
print("[4/9] Computing state transitions and saving plot...")
n_states = len(np.unique(all_states))
transition_counts = np.zeros((n_states, n_states), dtype=int)
for a, b in zip(all_states[:-1], all_states[1:]):
    transition_counts[a, b] += 1
transition_prob = transition_counts / transition_counts.sum(axis=1, keepdims=True)

plt.figure(figsize=(6, 5))
turquoise_cmap = LinearSegmentedColormap.from_list("white_turquoise", ["#FFFFFF", "#008080"])
im = plt.imshow(transition_prob, cmap=turquoise_cmap, interpolation='nearest')
plt.colorbar(im, label='Transition Probability')
plt.xlabel('To State')
plt.ylabel('From State')
plt.title('Predicted State Transition Probability Matrix')
plt.xticks(np.arange(n_states), [f'State {i}' for i in range(n_states)])
plt.yticks(np.arange(n_states), [f'State {i}' for i in range(n_states)])
for i in range(n_states):
    for j in range(n_states):
        plt.text(j, i, f"{transition_prob[i,j]:.2f}", ha='center', va='center',
                 color='black' if transition_prob[i,j]<0.5 else 'white', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'state_transition_matrix.pdf'))
plt.close()

# --------------------------------------------------------------------
# Trial-level data extraction
# --------------------------------------------------------------------
print(f"[5/9] Extracting trial-level data for {len(sessions)} sessions...")
all_trials = []
# Robust per-session block boundaries (last trial of each block), derived
# directly from the block-change marker (3091) -- independent of trial outcome.
session_block_ends = {}
for session_name in sessions:
    if session_name not in session_to_states:
        continue

    predicted_states = session_to_states[session_name]

    # Get RT from emissions
    rt_values = session_to_rt[session_name]
    if rt_values is not None:
        rt = rt_values[:len(predicted_states)]
    else:
        rt = np.full(len(predicted_states), np.nan)

    # Load log file for trial type info
    session_log_dir = os.path.join(raw_data_dir, session_name)
    if not os.path.isdir(session_log_dir):
        continue
    print(f"  Processing session {session_name}...")
    session_date_str = datetime.strptime(session_name, "%Y%m%d").strftime("%Y_%m_%d")
    log_files = [f for f in os.listdir(session_log_dir) if f.endswith('.log') and session_date_str in f]
    if not log_files:
        continue
    log_filepath = os.path.join(session_log_dir, log_files[0])
    with TextLog(log_filepath) as log:
        evt, ts, evt_desc, true_ts = log.parse_eventmarkers()
    trial_onset = ts[np.where(evt==3000)[0]]
    trial_end = ts[np.where(evt==3090)[0]]
    correct_idx = np.where(evt==1)[0]
    wrong_idx = np.where(evt==2)[0]
    miss_idx = np.where(np.isin(evt,[104,105,116,117,998]))[0]
    block_idx = np.where(evt==3091)[0]

    def find_trial(ts_event):
        trial_num = np.searchsorted(trial_onset, ts_event, side='right')-1
        if trial_num<0 or trial_num>=len(trial_end):
            return -1
        if trial_onset[trial_num]<=ts_event<=trial_end[trial_num]:
            return trial_num
        return -1

    block_end_trial_indices = sorted(set(np.searchsorted(trial_onset, ts[block_idx], side='right')-1))
    # Keep only in-range boundaries and store for the block-change analyses
    session_block_ends[session_name] = [b for b in block_end_trial_indices
                                        if 0 <= b < len(predicted_states)]
    with TextLog(log_filepath) as log:
        trial_data = log.get_info_per_trial(return_eventmarkers=True, return_loc=False)
    difficulty = np.array(['easy' if x in [30,70] else 'hard' if x in [49,51] else 'unknown' for x in trial_data['MorphTarget']])

    for i, state in enumerate(predicted_states):
        trial_type='other'
        if i in block_end_trial_indices:
            if i in [find_trial(ts[j]) for j in correct_idx]: trial_type='correct_end'
            elif i in [find_trial(ts[j]) for j in wrong_idx]: trial_type='wrong_end'
            elif i in [find_trial(ts[j]) for j in miss_idx]: trial_type='miss_end'
        else:
            if i in [find_trial(ts[j]) for j in correct_idx]: trial_type='correct'
            elif i in [find_trial(ts[j]) for j in wrong_idx]: trial_type='wrong'
            elif i in [find_trial(ts[j]) for j in miss_idx]: trial_type='miss'
        all_trials.append({'Session':session_name,'TrialIndex':i,'PredictedState':state,
                           'OriginalRT':rt[i],'TrialType':trial_type,
                           'Difficulty':difficulty[i] if i<len(difficulty) else 'unknown'})

trial_df = pd.DataFrame(all_trials)
trial_df['correct'] = trial_df['TrialType'].isin(['correct','correct_end'])
trial_df['wrong'] = trial_df['TrialType'].isin(['wrong','wrong_end'])
trial_df['misses'] = trial_df['TrialType'].isin(['miss','miss_end'])
trial_df['block_end'] = trial_df['TrialType'].str.endswith('_end')

# --------------------------------------------------------------------
# Trial outcome proportions per predicted state (matrix)
# --------------------------------------------------------------------
outcome_cols = ['correct', 'wrong', 'misses', 'block_end']
outcome_labels = ['Correct', 'Incorrect', 'Misses', 'Exit']
states = sorted(trial_df['PredictedState'].unique())
states = [s for s in states if s in states_to_include]

# ASR (adjusted standardized residual) per State x Outcome cell, with
# significance from a circular-shift permutation null (preserves the temporal
# autocorrelation of the sticky HMM state sequence within each session).
n_perms = 10000
rng = np.random.default_rng(42)
print(f"[6/9] Circular-shift ASR: State x Outcome ({n_perms} perms, {len(outcome_cols)} outcomes x {len(states)} states)...")

# Per-session ordered arrays shared by the trial-level analyses below.
trial_sorted = trial_df.sort_values(['Session', 'TrialIndex'])
state_flat = trial_sorted['PredictedState'].to_numpy()
session_flat = trial_sorted['Session'].to_numpy()
sess_slices = [np.where(session_flat == s)[0] for s in pd.unique(session_flat)]

outcome_arrays = [trial_sorted[c].to_numpy().astype(bool) for c in outcome_cols]
# asr_with_perm returns (n_states, n_outcomes); transpose to (outcomes, states)
obs_so, p_so = asr_with_perm(state_flat, sess_slices, states, outcome_arrays, n_perms, rng)
z_matrix = obs_so.T
p_matrix = p_so.T
sig_matrix = p_matrix < 0.05

# --- Diverging pastel light-purple to turquoise colormap ---
purple_turquoise = LinearSegmentedColormap.from_list(
    "purple_turquoise", ['#C8A2C8', '#FFFFFF', '#66CDAA'])

# --- Settings for Illustrator-safe PDF ---
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['savefig.transparent'] = False

# --- Plotting ---
z_abs_max = np.nanmax(np.abs(z_matrix))
fig, ax = plt.subplots(figsize=(8, 5))

for i_out in range(len(outcome_labels)):
    for j_st in range(len(states)):
        val = z_matrix[i_out, j_st]
        # Normalized color
        norm_val = (val / z_abs_max + 1) / 2 if not np.isnan(val) else 0.5
        color = purple_turquoise(norm_val)
        rect = plt.Rectangle((j_st - 0.5, i_out - 0.5), 1, 1,
                              facecolor=color, edgecolor='grey', linewidth=0.5)
        ax.add_patch(rect)
        label = f"{val:.1f}"
        if sig_matrix[i_out, j_st]:
            label += '*'
        ax.text(j_st, i_out, label, ha='center', va='center',
                color='black', fontsize=11, fontweight='bold')

ax.set_xlim(-0.5, len(states) - 0.5)
ax.set_ylim(-0.5, len(outcome_labels) - 0.5)
ax.set_xticks(np.arange(len(states)))
ax.set_xticklabels([f'{int(s)}' for s in states])
ax.set_xlabel('State')
ax.set_yticks(np.arange(len(outcome_labels)))
ax.set_yticklabels(outcome_labels)
ax.invert_yaxis()

# Colorbar
sm = plt.cm.ScalarMappable(cmap=purple_turquoise,
                            norm=plt.Normalize(vmin=-z_abs_max, vmax=z_abs_max))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, label='Z(w)')

plt.title('State × Outcome\n(* permutation p<0.05)')
plt.tight_layout()

output_file = os.path.join(output_dir, 'trial_outcome_vs_states_zscores.pdf')
plt.savefig(output_file, dpi=600, transparent=False, facecolor='white')
plt.close()

# --------------------------------------------------------------------
# Reaction Time analysis 
# --------------------------------------------------------------------
def extract_rt(val):
    if isinstance(val,(list,np.ndarray)):
        return float(val[0]) if len(val)>0 else np.nan
    return float(val)

valid_rt_df = trial_df.copy()
valid_rt_df['OriginalRT'] = valid_rt_df['OriginalRT'].apply(extract_rt)
valid_rt_df = valid_rt_df[valid_rt_df['OriginalRT'].notna()]
valid_rt_df = valid_rt_df[valid_rt_df['PredictedState'].isin(states_to_include)]

# Pairwise Mann-Whitney between states
pairwise_results=[]
for s1,s2 in combinations(states_to_include,2):
    rt1 = valid_rt_df.loc[valid_rt_df['PredictedState']==s1,'OriginalRT']
    rt2 = valid_rt_df.loc[valid_rt_df['PredictedState']==s2,'OriginalRT']
    if len(rt1)>0 and len(rt2)>0:
        stat,p=mannwhitneyu(rt1,rt2,alternative='two-sided')
    else:
        stat,p=np.nan,np.nan
    pairwise_results.append({'State1':s1,'State2':s2,'U-stat':stat,'p-value':p})
pairwise_df=pd.DataFrame(pairwise_results)
pairwise_df['p-corrected']=np.minimum(pairwise_df['p-value']*len(pairwise_df),1.0)
pairwise_df['significant']=pairwise_df['p-corrected']<0.05
pairwise_df.to_csv(os.path.join(output_dir,'rt_pairwise_comparisons.csv'),index=False)
print(pairwise_df)

# --- RT plot with custom colors ---
palette = [state_colors[s] for s in states_to_include]

plt.figure(figsize=(10,6))
sns.violinplot(data=valid_rt_df, x='PredictedState', y='OriginalRT', palette=palette, inner='box')
plt.xlabel('Predicted State'); plt.ylabel('Reaction Time (RT)')
plt.title('RT Distribution by Selected States')
y_max = valid_rt_df['OriginalRT'].max()
h = 0.05*y_max
for i,row in pairwise_df.iterrows():
    if row['significant']:
        x1,x2 = states_to_include.index(row['State1']), states_to_include.index(row['State2'])
        y = y_max + h*(i+1)
        plt.plot([x1,x1,x2,x2],[y,y+h,y+h,y],color='black')
        plt.text((x1+x2)/2,y+h*1.5,'*',ha='center')
plt.tight_layout()
plt.savefig(os.path.join(output_dir,'rt_by_state_selected.pdf'))
plt.close()

# ====================================================================
# Figure: State × Outcome × Difficulty Z-score heatmap
# ====================================================================
# Build outcome column for 3-way: correct, wrong, exit (block_end)
# Combined with difficulty (easy, hard)
outcome_diff_cols = []
outcome_diff_labels = []
for oc, ol in [('correct', 'correct'), ('block_end', 'exit'), ('wrong', 'wrong')]:
    for diff in ['easy', 'hard']:
        col_name = f'{oc}_{diff}'
        trial_df[col_name] = trial_df[oc] & (trial_df['Difficulty'] == diff)
        outcome_diff_cols.append(col_name)
        outcome_diff_labels.append(f'{ol} | {diff}')

print(f"[7/9] Circular-shift ASR: State x Outcome x Difficulty ({n_perms} perms, {len(outcome_diff_cols)} combos x {len(states)} states)...")
# Re-sort to align the newly-added outcome_diff columns to the shared
# per-session ordering (state_flat / sess_slices from the previous section).
trial_sorted_od = trial_df.sort_values(['Session', 'TrialIndex'])
od_arrays = [trial_sorted_od[c].to_numpy().astype(bool) for c in outcome_diff_cols]
obs_od, p_od = asr_with_perm(state_flat, sess_slices, states, od_arrays, n_perms, rng)
z_matrix_od = obs_od.T
p_matrix_od = p_od.T
sig_matrix_od = p_matrix_od < 0.05

z_abs_max_od = np.nanmax(np.abs(z_matrix_od))
fig, ax = plt.subplots(figsize=(8, 6))
for i_out in range(len(outcome_diff_labels)):
    for j_st in range(len(states)):
        val = z_matrix_od[i_out, j_st]
        norm_val = (val / z_abs_max_od + 1) / 2 if not np.isnan(val) else 0.5
        color = purple_turquoise(norm_val)
        rect = plt.Rectangle((j_st - 0.5, i_out - 0.5), 1, 1,
                              facecolor=color, edgecolor='grey', linewidth=0.5)
        ax.add_patch(rect)
        label = f"{val:.1f}"
        if sig_matrix_od[i_out, j_st]:
            label += '*'
        ax.text(j_st, i_out, label, ha='center', va='center',
                color='black', fontsize=11, fontweight='bold')

ax.set_xlim(-0.5, len(states) - 0.5)
ax.set_ylim(-0.5, len(outcome_diff_labels) - 0.5)
ax.set_xticks(np.arange(len(states)))
ax.set_xticklabels([f'{int(s)}' for s in states])
ax.set_xlabel('State')
ax.set_yticks(np.arange(len(outcome_diff_labels)))
ax.set_yticklabels(outcome_diff_labels)
ax.invert_yaxis()

sm = plt.cm.ScalarMappable(cmap=purple_turquoise,
                            norm=plt.Normalize(vmin=-z_abs_max_od, vmax=z_abs_max_od))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, label='Z(w)')
plt.title('State × Outcome × Difficulty\n(* permutation p<0.05)')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'trial_outcome_difficulty_vs_states_zscores.pdf'),
            dpi=600, transparent=False, facecolor='white')
plt.close()

# ====================================================================
# Figure: State Probabilities Across Normalized Block Position
# ====================================================================
# Assign each trial a normalized position within its block (0-100%)
n_bins = 10
bin_edges = np.linspace(0, 1, n_bins + 1)
bin_labels = [f'{int(bin_edges[i]*100)}-{int(bin_edges[i+1]*100)}%' for i in range(n_bins)]

# Identify blocks per session and assign normalized position
block_pos_records = []
for session_name in sessions:
    if session_name not in session_to_states:
        continue
    predicted_states = session_to_states[session_name]

    # Robust block boundaries (last trial of each block) from the 3091 marker:
    # every block change, independent of trial outcome.
    block_end_indices = np.array(sorted(session_block_ends.get(session_name, [])))
    # Build block boundaries: start at 0, end after each block_end
    block_starts = [0] + [be + 1 for be in block_end_indices if be + 1 < len(predicted_states)]
    block_ends = list(block_end_indices) + [len(predicted_states) - 1]
    # Pair starts and ends
    blocks = []
    for bs in block_starts:
        matching_ends = [be for be in block_ends if be >= bs]
        if matching_ends:
            blocks.append((bs, matching_ends[0]))

    for b_start, b_end in blocks:
        block_len = b_end - b_start + 1
        if block_len < 2:
            continue
        for t in range(b_start, b_end + 1):
            norm_pos = (t - b_start) / block_len
            bin_idx = min(int(norm_pos * n_bins), n_bins - 1)
            if t < len(predicted_states):
                block_pos_records.append({
                    'Session': session_name,
                    'T': t,  # absolute trial index, for circular-shift re-reads
                    'State': int(predicted_states[t]),
                    'NormBin': bin_idx,
                    'NormBinLabel': bin_labels[bin_idx]
                })

block_pos_df = pd.DataFrame(block_pos_records)

# ASR per (state, normalized-block-bin) cell, with circular-shift significance.
# Raveled, per-session state sequences are re-read at each record's absolute
# trial index after every shift (shared with the block-transition section).
state_seqs = {s: np.asarray(seq).ravel() for s, seq in session_to_states.items()}

bp_states = sorted(block_pos_df['State'].unique())
bp_states = [s for s in bp_states if s in states_to_include]

print(f"[8/9] Circular-shift ASR: State x Block Position ({n_perms} perms, {len(bp_states)} states x {n_bins} bins)...")
bin_arrays = [(block_pos_df['NormBin'] == j).to_numpy() for j in range(n_bins)]
z_block_pos, p_block_pos = asr_with_perm_records(
    block_pos_df['Session'].to_numpy(), block_pos_df['T'].to_numpy(),
    bin_arrays, bp_states, state_seqs, n_perms, rng)
sig_block_pos = p_block_pos < 0.05

z_abs_max_bp = np.nanmax(np.abs(z_block_pos))
if np.isnan(z_abs_max_bp) or z_abs_max_bp == 0:
    z_abs_max_bp = 1.0

fig, ax = plt.subplots(figsize=(14, 5))
for i_st in range(len(bp_states)):
    for j_bin in range(n_bins):
        val = z_block_pos[i_st, j_bin]
        norm_val = (val / z_abs_max_bp + 1) / 2 if not np.isnan(val) else 0.5
        color = purple_turquoise(norm_val)
        rect = plt.Rectangle((j_bin - 0.5, i_st - 0.5), 1, 1,
                              facecolor=color, edgecolor='grey', linewidth=0.5)
        ax.add_patch(rect)
        if not np.isnan(val):
            label = f"{val:.1f}"
            if sig_block_pos[i_st, j_bin]:
                label += '*'
            ax.text(j_bin, i_st, label, ha='center', va='center',
                    color='black', fontsize=9, fontweight='bold')

ax.set_xlim(-0.5, n_bins - 0.5)
ax.set_ylim(-0.5, len(bp_states) - 0.5)
ax.set_xticks(np.arange(n_bins))
ax.set_xticklabels(bin_labels, rotation=45, ha='right')
ax.set_xlabel('Normalized Position Within Block')
ax.set_yticks(np.arange(len(bp_states)))
ax.set_yticklabels([f'State {s}' for s in bp_states])
ax.set_ylabel('State')
ax.invert_yaxis()

sm = plt.cm.ScalarMappable(cmap=purple_turquoise,
                            norm=plt.Normalize(vmin=-z_abs_max_bp, vmax=z_abs_max_bp))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, label='Z(w)')
plt.title('State Probabilities Across Normalized Block Position\n(* permutation p<0.05)')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'state_prob_block_position_zscore.pdf'),
            dpi=600, transparent=False, facecolor='white')
plt.close()

# ====================================================================
# Figure: State Probabilities Around Block Transitions : Different from rest of windows
# ====================================================================
# Offset 0 = first trial of the new block, inclusive range, clipped to
# within-session bounds.
window_before = 8  # trials before transition
window_after = 5   # trials after transition
total_window = window_before + window_after

transition_records = []
for session_name in sessions:
    if session_name not in session_to_states:
        continue
    predicted_states = session_to_states[session_name]
    # Robust, outcome-independent block boundaries (3091 marker).
    block_end_indices = np.array(sorted(session_block_ends.get(session_name, [])))

    for be in block_end_indices:
        # Transition point is at be+1 (first trial of new block)
        trans_point = be + 1
        for offset in range(-window_before, window_after + 1):
            t = trans_point + offset
            if 0 <= t < len(predicted_states):
                transition_records.append({
                    'Session': session_name,
                    'T': t,  # absolute trial index, for circular-shift re-reads
                    'State': int(predicted_states[t]),
                    'Offset': offset
                })

trans_df = pd.DataFrame(transition_records)
offsets = sorted(trans_df['Offset'].unique())

tr_states = sorted(trans_df['State'].unique())
tr_states = [s for s in tr_states if s in states_to_include]

print(f"[9/9] Circular-shift ASR: State x Block Transitions ({n_perms} perms, {len(tr_states)} states x {len(offsets)} offsets)...")
offset_arrays = [(trans_df['Offset'] == o).to_numpy() for o in offsets]
z_trans, p_trans = asr_with_perm_records(
    trans_df['Session'].to_numpy(), trans_df['T'].to_numpy(),
    offset_arrays, tr_states, state_seqs, n_perms, rng)
sig_trans = p_trans < 0.05

z_abs_max_tr = np.nanmax(np.abs(z_trans))
if np.isnan(z_abs_max_tr) or z_abs_max_tr == 0:
    z_abs_max_tr = 1.0

fig, ax = plt.subplots(figsize=(14, 5))
for i_st in range(len(tr_states)):
    for j_off in range(len(offsets)):
        val = z_trans[i_st, j_off]
        norm_val = (val / z_abs_max_tr + 1) / 2 if not np.isnan(val) else 0.5
        color = purple_turquoise(norm_val)
        rect = plt.Rectangle((j_off - 0.5, i_st - 0.5), 1, 1,
                              facecolor=color, edgecolor='grey', linewidth=0.5)
        ax.add_patch(rect)
        if not np.isnan(val):
            label = f"{val:.1f}"
            if sig_trans[i_st, j_off]:
                label += '*'
            ax.text(j_off, i_st, label, ha='center', va='center',
                    color='black', fontsize=8, fontweight='bold')

# Add dashed line at transition point (offset=0)
trans_x = offsets.index(0) if 0 in offsets else None
if trans_x is not None:
    ax.axvline(x=trans_x, color='black', linestyle='--', linewidth=2)

ax.set_xlim(-0.5, len(offsets) - 0.5)
ax.set_ylim(-0.5, len(tr_states) - 0.5)
ax.set_xticks(np.arange(len(offsets)))
ax.set_xticklabels([str(o) for o in offsets])
ax.set_xlabel('Trial Position Relative to Block Transition')
ax.set_yticks(np.arange(len(tr_states)))
ax.set_yticklabels([f'State {s}' for s in tr_states])
ax.set_ylabel('State')
ax.invert_yaxis()

sm = plt.cm.ScalarMappable(cmap=purple_turquoise,
                            norm=plt.Normalize(vmin=-z_abs_max_tr, vmax=z_abs_max_tr))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, label='Z(w)')
plt.title('State Probabilities Around Block Transitions\n(* permutation p<0.05)')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'state_prob_block_transitions_zscore.pdf'),
            dpi=600, transparent=False, facecolor='white')
plt.close()

# ====================================================================
# Figure: State Probabilities Around Block Transitions (Normalized to Baseline)
# ====================================================================
# This is a MAGNITUDE (effect-size) panel, so it uses the SOFT posteriors
# rather than hard labels. For each (state, offset):
#     ratio = mean P(state | trial) over records at that offset
#             ------------------------------------------------------
#                 overall mean P(state | trial) across all trials
# A ratio of 1.0 means "as common as its overall rate"; 2.0 means twice as
# common. Averaging the graded posterior (instead of hard 0/1 membership)
# keeps the rare states (1 and 3) from being swamped by label noise.
baseline_soft = state_probs.mean(axis=0)  # (K,) overall mean posterior per state
trans_probs = np.array([session_to_probs[s][t]
                        for s, t in zip(trans_df['Session'], trans_df['T'])])  # (n_records, K)
ratio_trans = np.full((len(tr_states), len(offsets)), np.nan)
for j_off, offset in enumerate(offsets):
    mask = (trans_df['Offset'] == offset).to_numpy()
    if mask.sum() < 5:
        continue
    mean_p = trans_probs[mask].mean(axis=0)  # (K,) mean posterior at this offset
    for i_st, state in enumerate(tr_states):
        if baseline_soft[state] > 0:
            ratio_trans[i_st, j_off] = mean_p[state] / baseline_soft[state]

fig, ax = plt.subplots(figsize=(14, 5))
# Same diverging colormap as the other panels: purple = below baseline,
# white = baseline (1.0), turquoise = above baseline. Centered at 1.0.
vmax = max(np.nanmax(ratio_trans), 2.0)
im = ax.imshow(ratio_trans, aspect='auto', cmap=purple_turquoise,
               norm=TwoSlopeNorm(vmin=0, vcenter=1.0, vmax=vmax),
               interpolation='nearest')

# Add dashed line at transition point (offset=0)
trans_x = offsets.index(0) if 0 in offsets else None
if trans_x is not None:
    ax.axvline(x=trans_x, color='black', linestyle='--', linewidth=2)

ax.set_xticks(np.arange(len(offsets)))
ax.set_xticklabels([str(o) for o in offsets])
ax.set_xlabel('Trial Position Relative to Block Transition')
ax.set_yticks(np.arange(len(tr_states)))
ax.set_yticklabels([f'State {s}' for s in tr_states])
ax.set_ylabel('State')

cbar = plt.colorbar(im, ax=ax, label='Relative to Baseline (1.0 = expected)')
plt.title('Soft State Probability Around Block Transitions (Normalized to Baseline)')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'state_prob_block_transitions_baseline_ratio.pdf'),
            dpi=600, transparent=False, facecolor='white')
plt.close()

# ====================================================================
# Figure: State Probabilities Across Normalized Block Position (Normalized to Baseline)
# ====================================================================
# Soft-probability magnitude counterpart of the block-position ASR panel above.
# For each (state, normalized-position bin):
#     ratio = mean P(state | trial) over trials in that bin / overall mean P(state)
bp_probs = np.array([session_to_probs[s][t]
                     for s, t in zip(block_pos_df['Session'], block_pos_df['T'])])  # (n_records, K)
ratio_block_pos = np.full((len(bp_states), n_bins), np.nan)
for j_bin in range(n_bins):
    mask = (block_pos_df['NormBin'] == j_bin).to_numpy()
    if mask.sum() < 5:
        continue
    mean_p = bp_probs[mask].mean(axis=0)  # (K,) mean posterior in this bin
    for i_st, state in enumerate(bp_states):
        if baseline_soft[state] > 0:
            ratio_block_pos[i_st, j_bin] = mean_p[state] / baseline_soft[state]

fig, ax = plt.subplots(figsize=(14, 5))
# Same diverging colormap as the other panels: purple = below baseline,
# white = baseline (1.0), turquoise = above baseline. Centered at 1.0.
vmax = max(np.nanmax(ratio_block_pos), 2.0)
im = ax.imshow(ratio_block_pos, aspect='auto', cmap=purple_turquoise,
               norm=TwoSlopeNorm(vmin=0, vcenter=1.0, vmax=vmax),
               interpolation='nearest')
ax.set_xticks(np.arange(n_bins))
ax.set_xticklabels(bin_labels, rotation=45, ha='right')
ax.set_xlabel('Normalized Position Within Block')
ax.set_yticks(np.arange(len(bp_states)))
ax.set_yticklabels([f'State {s}' for s in bp_states])
ax.set_ylabel('State')
cbar = plt.colorbar(im, ax=ax, label='Relative to Baseline (1.0 = expected)')
plt.title('Soft State Probability Across Normalized Block Position (Normalized to Baseline)')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'state_prob_block_position_baseline_ratio.pdf'),
            dpi=600, transparent=False, facecolor='white')
plt.close()

print("Done! All plots saved.")
