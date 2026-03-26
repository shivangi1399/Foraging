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
from matplotlib.colors import LinearSegmentedColormap
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
# User Config
# --------------------------------------------------------------------
N_STATES_TO_USE = 4
states_to_include = list(range(N_STATES_TO_USE))

# --------------------------------------------------------------------
# Load states info for all sessions (from centralized state assignments)
# --------------------------------------------------------------------
print("[1/9] Loading state assignments and session index...")
state_assignments = np.load(f'{states_data_dir}/foraging_shivangi_no_sess1_clipped_state_assignments.npy')
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

session_to_states = {}
for sess in session_index:
    session_id = sess['session_id']
    session_date = session_id.split('_')[1]
    session_to_states[session_date] = state_assignments[
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

# Compute Z(w) scores: each state vs all other states, per outcome
n_perms = 10000
rng = np.random.default_rng(42)
print(f"[6/9] Permutation tests: State x Outcome ({n_perms} perms, {len(outcome_cols)} outcomes x {len(states)} states)...")
z_matrix = np.full((len(outcome_labels), len(states)), np.nan)
p_matrix = np.full((len(outcome_labels), len(states)), np.nan)

for i_out, col in enumerate(outcome_cols):
    print(f"  Outcome: {outcome_labels[i_out]}...")
    for j_st, state in enumerate(states):
        in_state = trial_df.loc[trial_df['PredictedState']==state, col].values.astype(float)
        out_state = trial_df.loc[trial_df['PredictedState']!=state, col].values.astype(float)
        if len(in_state) < 2 or len(out_state) < 2:
            continue
        u_stat, p_val = mannwhitneyu(in_state, out_state, alternative='two-sided')
        n1, n2 = len(in_state), len(out_state)
        mu = n1 * n2 / 2
        sigma = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
        z = (u_stat - mu) / sigma
        # Sign: positive = state has MORE of this outcome than rest
        z_matrix[i_out, j_st] = z

        # Permutation test for significance
        all_vals = np.concatenate([in_state, out_state])
        count_extreme = 0
        for _ in range(n_perms):
            rng.shuffle(all_vals)
            perm_in = all_vals[:n1]
            perm_out = all_vals[n1:]
            u_perm, _ = mannwhitneyu(perm_in, perm_out, alternative='two-sided')
            z_perm = (u_perm - mu) / sigma
            if abs(z_perm) >= abs(z):
                count_extreme += 1
        p_matrix[i_out, j_st] = count_extreme / n_perms

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

print(f"[7/9] Permutation tests: State x Outcome x Difficulty ({n_perms} perms, {len(outcome_diff_cols)} combos x {len(states)} states)...")
z_matrix_od = np.full((len(outcome_diff_labels), len(states)), np.nan)
p_matrix_od = np.full((len(outcome_diff_labels), len(states)), np.nan)

for i_out, col in enumerate(outcome_diff_cols):
    print(f"  {outcome_diff_labels[i_out]}...")
    for j_st, state in enumerate(states):
        in_state = trial_df.loc[trial_df['PredictedState'] == state, col].values.astype(float)
        out_state = trial_df.loc[trial_df['PredictedState'] != state, col].values.astype(float)
        if len(in_state) < 2 or len(out_state) < 2:
            continue
        u_stat, _ = mannwhitneyu(in_state, out_state, alternative='two-sided')
        n1, n2 = len(in_state), len(out_state)
        mu = n1 * n2 / 2
        sigma = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
        z = (u_stat - mu) / sigma
        z_matrix_od[i_out, j_st] = z

        all_vals = np.concatenate([in_state, out_state])
        count_extreme = 0
        for _ in range(n_perms):
            rng.shuffle(all_vals)
            perm_in = all_vals[:n1]
            perm_out = all_vals[n1:]
            u_perm, _ = mannwhitneyu(perm_in, perm_out, alternative='two-sided')
            z_perm = (u_perm - mu) / sigma
            if abs(z_perm) >= abs(z):
                count_extreme += 1
        p_matrix_od[i_out, j_st] = count_extreme / n_perms

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

    # Find block boundaries from trial_df
    sess_df = trial_df[trial_df['Session'] == session_name].sort_values('TrialIndex')
    block_end_indices = sess_df[sess_df['block_end']].TrialIndex.values
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
                    'State': int(predicted_states[t]),
                    'NormBin': bin_idx,
                    'NormBinLabel': bin_labels[bin_idx]
                })

block_pos_df = pd.DataFrame(block_pos_records)

# Compute observed vs expected (baseline) probability per state per bin
# Baseline = overall state probability
state_baseline = block_pos_df['State'].value_counts(normalize=True)

# Z-score approach: for each (state, bin), compare proportion vs shuffled
bp_states = sorted(block_pos_df['State'].unique())
bp_states = [s for s in bp_states if s in states_to_include]

print(f"[8/9] Permutation tests: State x Block Position ({n_perms} perms, {len(bp_states)} states x {n_bins} bins)...")
z_block_pos = np.full((len(bp_states), n_bins), np.nan)
p_block_pos = np.full((len(bp_states), n_bins), np.nan)

for i_st, state in enumerate(bp_states):
    print(f"  State {state} ({i_st+1}/{len(bp_states)})...")
    for j_bin in range(n_bins):
        in_bin = block_pos_df[block_pos_df['NormBin'] == j_bin]
        n_total = len(in_bin)
        if n_total < 5:
            continue
        observed = (in_bin['State'] == state).sum()
        obs_prop = observed / n_total

        # Permutation test: shuffle state labels within each bin
        all_states_bin = in_bin['State'].values.copy()
        obs_diff = obs_prop - state_baseline.get(state, 0)
        count_extreme = 0
        for _ in range(n_perms):
            rng.shuffle(all_states_bin)
            perm_prop = (all_states_bin == state).sum() / n_total
            perm_diff = perm_prop - state_baseline.get(state, 0)
            if abs(perm_diff) >= abs(obs_diff):
                count_extreme += 1
        p_block_pos[i_st, j_bin] = count_extreme / n_perms

        # Z-score via Mann-Whitney: is state membership different in this bin?
        is_state = (block_pos_df['State'] == state).astype(float).values
        is_bin = (block_pos_df['NormBin'] == j_bin).astype(float).values
        in_group = is_state[is_bin == 1]
        out_group = is_state[is_bin == 0]
        if len(in_group) >= 2 and len(out_group) >= 2:
            u_stat, _ = mannwhitneyu(in_group, out_group, alternative='two-sided')
            n1, n2 = len(in_group), len(out_group)
            mu = n1 * n2 / 2
            sigma = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
            z_block_pos[i_st, j_bin] = (u_stat - mu) / sigma

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
window_before = 9  # trials before transition
window_after = 6   # trials after transition
total_window = window_before + window_after

transition_records = []
for session_name in sessions:
    if session_name not in session_to_states:
        continue
    predicted_states = session_to_states[session_name]
    sess_df = trial_df[trial_df['Session'] == session_name].sort_values('TrialIndex')
    block_end_indices = sess_df[sess_df['block_end']].TrialIndex.values

    for be in block_end_indices:
        # Transition point is at be+1 (first trial of new block)
        trans_point = be + 1
        for offset in range(-window_before, window_after):
            t = trans_point + offset
            if 0 <= t < len(predicted_states):
                transition_records.append({
                    'Session': session_name,
                    'State': int(predicted_states[t]),
                    'Offset': offset
                })

trans_df = pd.DataFrame(transition_records)
offsets = sorted(trans_df['Offset'].unique())

tr_states = sorted(trans_df['State'].unique())
tr_states = [s for s in tr_states if s in states_to_include]

print(f"[9/9] Permutation tests: State x Block Transitions ({n_perms} perms, {len(tr_states)} states x {len(offsets)} offsets)...")
z_trans = np.full((len(tr_states), len(offsets)), np.nan)
p_trans = np.full((len(tr_states), len(offsets)), np.nan)

# Baseline state proportions from full dataset
state_baseline_trans = trial_df['PredictedState'].value_counts(normalize=True)

for i_st, state in enumerate(tr_states):
    print(f"  State {state} ({i_st+1}/{len(tr_states)})...")
    for j_off, offset in enumerate(offsets):
        in_offset = trans_df[trans_df['Offset'] == offset]
        n_total = len(in_offset)
        if n_total < 5:
            continue
        observed = (in_offset['State'] == state).sum()
        obs_prop = observed / n_total
        obs_diff = obs_prop - state_baseline_trans.get(state, 0)

        # Permutation test
        all_states_off = in_offset['State'].values.copy()
        count_extreme = 0
        for _ in range(n_perms):
            rng.shuffle(all_states_off)
            perm_prop = (all_states_off == state).sum() / n_total
            perm_diff = perm_prop - state_baseline_trans.get(state, 0)
            if abs(perm_diff) >= abs(obs_diff):
                count_extreme += 1
        p_trans[i_st, j_off] = count_extreme / n_perms

        # Z-score via Mann-Whitney
        is_state = (trans_df['State'] == state).astype(float).values
        is_offset = (trans_df['Offset'] == offset).astype(float).values
        in_group = is_state[is_offset == 1]
        out_group = is_state[is_offset == 0]
        if len(in_group) >= 2 and len(out_group) >= 2:
            u_stat, _ = mannwhitneyu(in_group, out_group, alternative='two-sided')
            n1, n2 = len(in_group), len(out_group)
            mu = n1 * n2 / 2
            sigma = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
            z_trans[i_st, j_off] = (u_stat - mu) / sigma

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
# For each (state, offset): observed_proportion / baseline_proportion
# Baseline = overall state probability from full dataset
ratio_trans = np.full((len(tr_states), len(offsets)), np.nan)
for i_st, state in enumerate(tr_states):
    baseline_p = state_baseline_trans.get(state, 0)
    if baseline_p == 0:
        continue
    for j_off, offset in enumerate(offsets):
        in_offset = trans_df[trans_df['Offset'] == offset]
        n_total = len(in_offset)
        if n_total < 5:
            continue
        obs_prop = (in_offset['State'] == state).sum() / n_total
        ratio_trans[i_st, j_off] = obs_prop / baseline_p

fig, ax = plt.subplots(figsize=(14, 5))
vmax = np.nanmax(ratio_trans)
im = ax.imshow(ratio_trans, aspect='auto', cmap='RdBu_r', vmin=0, vmax=max(vmax, 2.0),
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
plt.title('State Probabilities Around Block Transitions (Normalized to Baseline)')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'state_prob_block_transitions_baseline_ratio.pdf'),
            dpi=600, transparent=False, facecolor='white')
plt.close()

print("Done! All plots saved.")
