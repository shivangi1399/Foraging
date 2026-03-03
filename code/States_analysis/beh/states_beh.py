import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd
from datetime import datetime
import seaborn as sns
from itertools import groupby
from scipy.stats import zscore

# Custom module paths
sys.path.insert(1, '/mnt/cs/projects/MWzeronoise/Analysis/4Shivangi/code/functions/unreal_logfile')
from parse_logfile import TextLog
from preprocessing import align_ephys as align
from preprocessing import snippet_ephys as snip
import syncopy as spy

# Paths
input_dir = '/cs/projects/MWzeronoise/Analysis/4Shivangi/Datasets/states_analysis'
output_dir = '/cs/projects/MWzeronoise/Analysis/4Shivangi/plots/states_analysis'
raw_data_dir = '/cs/projects/MWzeronoise/Analysis/4Shivangi/Datasets/raw_data'

# Analysis of predicted states --------------------------------------------------------------------------------
sessions = ['20230203', '20230206', '20230207', '20230208', '20230209', '20230213', '20230214']

os.makedirs(output_dir, exist_ok=True)
os.chdir(input_dir)
npz_files = [f for f in os.listdir('.') if f.endswith('.npz') and any(sess in f for sess in sessions)]

all_states = []
for fname in npz_files:
    session_name = next((sess for sess in sessions if sess in fname), None)
    if session_name is None:
        print(f"No matching session found for {fname}")
        continue

    session_output_dir = os.path.join(output_dir, session_name)
    os.makedirs(session_output_dir, exist_ok=True)

    data = np.load(fname)
    predicted_states = data['predicted_states']
    state_sequence = predicted_states.ravel().tolist()
    all_states.extend(state_sequence)

# State durations ----
state_lengths = []
state_ids = []

for state, group in groupby(all_states):
    length = len(list(group))
    state_lengths.append(length)
    state_ids.append(state)

df_state_lengths = pd.DataFrame({'state': state_ids, 'length': state_lengths})

plt.figure(figsize=(8, 4))
sns.histplot(data=df_state_lengths, x='length', hue='state', multiple='stack', bins=30)
plt.title("State Durations Across All Sessions")
plt.xlabel("State Duration (trials)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'state_duration.png'))
plt.close()

# State Transitions ----
n_states = len(np.unique(all_states))
transition_counts = np.zeros((n_states, n_states), dtype=int)
for (a, b) in zip(all_states[:-1], all_states[1:]):
    transition_counts[a, b] += 1
transition_prob = transition_counts / transition_counts.sum(axis=1, keepdims=True)

plt.figure(figsize=(6, 5))
im = plt.imshow(transition_prob, cmap='Blues', interpolation='nearest')
plt.colorbar(im, label='Transition Probability')
plt.xlabel('To State')
plt.ylabel('From State')
plt.title('Predicted State Transition Probability Matrix')
plt.xticks(np.arange(n_states), [f'State {i}' for i in range(n_states)])
plt.yticks(np.arange(n_states), [f'State {i}' for i in range(n_states)])

for i in range(n_states):
    for j in range(n_states):
        plt.text(j, i, f"{transition_prob[i, j]:.2f}", ha='center', va='center',
                    color='black' if transition_prob[i, j] < 0.5 else 'white', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'state_transition_matrix.png'))
plt.close()

# What do each of the states represent --------------------------------------------------------------------

all_trials = []

for fname in npz_files:
    fname
    session_name = next((sess for sess in sessions if sess in fname), None)
    session_log_dir = os.path.join(raw_data_dir, session_name)
    if not os.path.isdir(session_log_dir):
        print(f"Directory does not exist for session: {session_name}")
        continue

    session_date_str = datetime.strptime(session_name, "%Y%m%d").strftime("%Y_%m_%d")

    log_files = [f for f in os.listdir(session_log_dir) if f.endswith('.log') and session_date_str in f]

    if not log_files:
        print(f"No log file found for session: {session_name}")
        continue

    log_filepath = os.path.join(session_log_dir, log_files[0])

    with TextLog(log_filepath) as log:
        start_trial_times = log.parse_all_state_times(state='StartTrial', times='StateStarted')
        end_trial_times = log.parse_all_state_times(state='EndTrial', times='StateStarted')
        evt, ts, evt_desc, true_ts = log.parse_eventmarkers()

    trial_onset = ts[np.where(evt == 3000)[0]]
    trial_end = ts[np.where(evt == 3090)[0]]
    correct_idx = np.where(evt == 1)[0]
    wrong_idx = np.where(evt == 2)[0]
    miss_idx = np.where((evt == 117) | (evt == 116) | (evt == 104) | (evt == 105) | (evt == 998))[0]
    block_idx = np.where(evt == 3091)[0]

    def find_trial(ts_event):
        trial_num = np.searchsorted(trial_onset, ts_event, side='right') - 1
        if trial_num < 0 or trial_num >= len(trial_end):
            return -1
        if trial_onset[trial_num] <= ts_event <= trial_end[trial_num]:
            return trial_num
        return -1

    block_end_trial_indices = sorted(set(
        np.searchsorted(trial_onset, ts[block_idx], side='right') - 1
    ))

    with TextLog(log_filepath) as log:
        trial_data = log.get_info_per_trial(return_eventmarkers=True, return_loc=False)

    difficulty = np.array(['easy' if x in [30, 70] else 'hard' if x in [49, 51] else 'unknown' for x in trial_data['MorphTarget']])

    predicted_states = np.load(fname)['predicted_states']
    rt = np.load(fname)['original_emissions'] if 'original_emissions' in np.load(fname).files else np.full_like(predicted_states, np.nan, dtype=float)

    for i, state in enumerate(predicted_states):
        trial_type = 'other'
        if i in block_end_trial_indices:
            if i in [find_trial(ts[j]) for j in correct_idx]: trial_type = 'correct_end'
            elif i in [find_trial(ts[j]) for j in wrong_idx]: trial_type = 'wrong_end'
            elif i in [find_trial(ts[j]) for j in miss_idx]: trial_type = 'miss_end'
        else:
            if i in [find_trial(ts[j]) for j in correct_idx]: trial_type = 'correct'
            elif i in [find_trial(ts[j]) for j in wrong_idx]: trial_type = 'wrong'
            elif i in [find_trial(ts[j]) for j in miss_idx]: trial_type = 'miss'

        all_trials.append({
            'Session': session_name,
            'TrialIndex': i,
            'PredictedState': state,
            'OriginalRT': rt[i],
            'TrialType': trial_type,
            'Difficulty': difficulty[i] if i < len(difficulty) else 'unknown'
        })

# Convert to DataFrame
trial_df = pd.DataFrame(all_trials)
trial_df['correct'] = trial_df['TrialType'].isin(['correct', 'correct_end'])
trial_df['wrong'] = trial_df['TrialType'].isin(['wrong', 'wrong_end'])
trial_df['misses'] = trial_df['TrialType'].isin(['miss', 'miss_end'])
trial_df['block_end'] = trial_df['TrialType'].str.endswith('_end')
print(trial_df.head(4))

# Group and summarize
summary = trial_df.groupby('PredictedState').agg(
    mean_rt=('OriginalRT', 'mean'),
    std_rt=('OriginalRT', 'std'),
    n_trials=('OriginalRT', 'count'),
    correct_rate=('correct', 'mean'),
    incorrect_rate=('wrong', 'mean'),
    miss_rate=('misses', 'mean'),
    block_end_rate=('block_end', 'mean')
).reset_index()

# Trial outcome vs states ----
states = summary['PredictedState']
x = np.arange(len(states))

matrix = np.vstack([
    summary['correct_rate'],
    summary['incorrect_rate'],
    summary['miss_rate'],
    summary['block_end_rate']
])

fig, ax = plt.subplots(figsize=(8, 5))
im = ax.imshow(matrix, cmap='Blues', aspect='auto')
ax.set_xticks(x)
ax.set_xticklabels([f'State {int(s)}' for s in states])
ax.set_yticks([0, 1, 2, 3])
ax.set_yticklabels(['Correct', 'Incorrect', 'Misses', 'Block End'])
for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        ax.text(j, i, f"{matrix[i, j]:.2f}", ha='center', va='center',
                color='black' if matrix[i, j] < 0.5 else 'white', fontsize=11)

plt.title('Trial Outcome Proportions by Predicted State')
plt.colorbar(im, ax=ax, label='Proportion')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'trial_outcome_vs_states.png'))
plt.close()

# Reaction Time plots ----
# Ensure OriginalRT is a float column 
def extract_rt(val):
    if isinstance(val, (list, np.ndarray)):
        return float(val[0]) if len(val) > 0 else np.nan
    return float(val)

valid_rt_df = trial_df.copy()
valid_rt_df['OriginalRT'] = valid_rt_df['OriginalRT'].apply(extract_rt)
valid_rt_df = valid_rt_df[valid_rt_df['OriginalRT'].notna()]

# RT by predicted state
if not valid_rt_df.empty:
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=valid_rt_df, x='PredictedState', y='OriginalRT', palette='Set2', inner='box')
    plt.xlabel('Predicted State')
    plt.ylabel('Reaction Time (RT)')
    plt.title('RT Distribution by Predicted State')
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rt_by_state.png'))
    plt.close()

# RT by state and difficulty
if not valid_rt_df.empty:
    plt.figure(figsize=(12, 6))
    sns.violinplot(data=valid_rt_df, x='PredictedState', y='OriginalRT', hue='Difficulty', split=True, palette='muted', inner='quartile')
    plt.xlabel('Predicted State')
    plt.ylabel('Reaction Time (RT)')
    plt.title('RT by Predicted State and Difficulty')
    plt.legend(title='Difficulty', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rt_by_state_difficulty.png'))
    plt.close()

# Faceted plot: RT by state per session
valid_rt_facet = valid_rt_df[valid_rt_df['Session'].notna()]
if not valid_rt_facet.empty:
    g = sns.FacetGrid(valid_rt_facet, col='Session', col_wrap=4, height=4, sharey=False)
    g.map_dataframe(lambda data, color: sns.boxplot(data=data, x='PredictedState', y='OriginalRT', palette='coolwarm'))
    g.map_dataframe(lambda data, color: sns.stripplot(data=data, x='PredictedState', y='OriginalRT', color='black', alpha=0.2, jitter=0.25))
    g.set_titles(col_template='Session {col_name}')
    for ax in g.axes.flatten():
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    g.fig.suptitle('RT by Predicted State per Session', y=1.03)
    g.tight_layout()
    g.savefig(os.path.join(output_dir, 'rt_by_state_per_session.png'))
    plt.close()

# State count around block changes ----
trial_df_copy = trial_df.copy()
block_change_indices = trial_df_copy.index[trial_df_copy['block_end']].tolist()# Get block change trial indices

window = 5 # Window of trials around each block change
offset_range = range(-window, window + 1)
state_counts = {offset: {s: 0 for s in range(4)} for offset in offset_range}

for idx in block_change_indices: # Count states at each offset
    for offset in offset_range:
        trial_idx = idx + offset
        if 0 <= trial_idx < len(trial_df_copy):
            state = trial_df_copy.iloc[trial_idx]['PredictedState']
            if state in state_counts[offset]:
                state_counts[offset][state] += 1

count_df = pd.DataFrame(state_counts).T.sort_index()  # shape: (offsets) x (states)

z_df = count_df.apply(zscore)  # z-score across offsets for each state

plt.figure(figsize=(10, 6))
for state in range(4):
    plt.plot(z_df.index, z_df[state], label=f'State {state}', marker='o')

plt.axvline(0, color='red', linestyle='--', label='Block Change')
plt.xlabel('Trials Relative to Block Change')
plt.ylabel('Z-scored Count')
plt.title('Normalized State Activity Around Block Change')
plt.xticks(list(offset_range))
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'state_count_block_change.png'))
plt.close()

# --------------------------------------------------------------------
# RT distribution normality & pairwise comparisons --------

import scipy.stats as stats
from itertools import combinations

# Only use valid RT data (from earlier in your script)
pooled_rt_df = valid_rt_df.copy()

# Skip state 4 if present
pooled_rt_df = pooled_rt_df[pooled_rt_df['PredictedState'] != 4]

# 1️⃣ Normality check (Shapiro–Wilk test) for each state
normality_results = []
for state, group in pooled_rt_df.groupby('PredictedState'):
    if len(group) > 3:  # Shapiro test requires >3 samples
        stat, p = stats.shapiro(group['OriginalRT'])
        normality_results.append({'State': state, 'W-statistic': stat, 'p-value': p})
    else:
        normality_results.append({'State': state, 'W-statistic': np.nan, 'p-value': np.nan})

normality_df = pd.DataFrame(normality_results)
normality_df.to_csv(os.path.join(output_dir, 'rt_normality_by_state.csv'), index=False)
print("\n=== Normality Test (Shapiro–Wilk) per State ===")
print(normality_df)

# 2️⃣ Pairwise comparisons between states
# Use non-parametric Mann–Whitney U test (robust for non-normal distributions)
pairwise_results = []
states = sorted(pooled_rt_df['PredictedState'].unique())

for s1, s2 in combinations(states, 2):
    rt1 = pooled_rt_df.loc[pooled_rt_df['PredictedState'] == s1, 'OriginalRT']
    rt2 = pooled_rt_df.loc[pooled_rt_df['PredictedState'] == s2, 'OriginalRT']
    if len(rt1) > 3 and len(rt2) > 3:
        stat, p = stats.mannwhitneyu(rt1, rt2, alternative='two-sided')
        pairwise_results.append({'State1': s1, 'State2': s2, 'U-statistic': stat, 'p-value': p})
    else:
        pairwise_results.append({'State1': s1, 'State2': s2, 'U-statistic': np.nan, 'p-value': np.nan})

pairwise_df = pd.DataFrame(pairwise_results)

# Bonferroni correction for multiple comparisons
pairwise_df['p-value_corrected'] = np.minimum(pairwise_df['p-value'] * len(pairwise_df), 1.0)
pairwise_df['significant'] = pairwise_df['p-value_corrected'] < 0.05

pairwise_df.to_csv(os.path.join(output_dir, 'rt_pairwise_comparisons.csv'), index=False)

print("\n=== Pairwise RT Differences Between States (Mann–Whitney U) ===")
print(pairwise_df)



# Normality visualization for RTs by predicted state --------

plot_dir = os.path.join(output_dir, 'normality_plots')
os.makedirs(plot_dir, exist_ok=True)

# Skip state 4 if present
states_to_plot = sorted([s for s in pooled_rt_df['PredictedState'].unique() if s != 4])

for state in states_to_plot:
    rt_vals = pooled_rt_df.loc[pooled_rt_df['PredictedState'] == state, 'OriginalRT'].dropna()

    if len(rt_vals) < 5:
        continue  # skip if too few samples

    # Fit normal distribution to data
    mu, sigma = np.mean(rt_vals), np.std(rt_vals)

    # --- 1️⃣ Histogram with fitted normal curve ---
    plt.figure(figsize=(6, 4))
    sns.histplot(rt_vals, kde=False, stat='density', bins=20, color='skyblue')
    x_vals = np.linspace(rt_vals.min(), rt_vals.max(), 100)
    plt.plot(x_vals, stats.norm.pdf(x_vals, mu, sigma), 'r--', lw=2, label='Normal fit')
    plt.title(f'State {state}: RT Distribution\n(mean={mu:.2f}, σ={sigma:.2f})')
    plt.xlabel('Reaction Time (RT)')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'rt_hist_normalfit_state{state}.pdf'))
    plt.close()

    # --- 2️⃣ Q-Q Plot ---
    plt.figure(figsize=(5, 5))
    stats.probplot(rt_vals, dist="norm", plot=plt)
    plt.title(f'State {state}: Q-Q Plot')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'rt_qqplot_state{state}.pdf'))
    plt.close()

print(f"\nNormality plots saved in: {plot_dir}")
