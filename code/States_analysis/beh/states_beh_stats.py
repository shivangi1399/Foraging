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
from itertools import combinations
from matplotlib.colors import LinearSegmentedColormap
import json

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
input_dir = '/cs/projects/MWzeronoise/Analysis/4Shivangi/Datasets/states_analysis/old_RT'
output_dir = '/cs/projects/MWzeronoise/Analysis/4Shivangi/plots/states_analysis'
raw_data_dir = '/cs/projects/MWzeronoise/Analysis/4Shivangi/Datasets/raw_data'
os.makedirs(output_dir, exist_ok=True)
os.chdir(input_dir)

# --------------------------------------------------------------------
# User selection: which states to include
# --------------------------------------------------------------------
states_to_include = [0, 1, 2, 3]  # select here for RT & block-change analysis

# --------------------------------------------------------------------
# Sessions & load predicted states
# --------------------------------------------------------------------
sessions = ['20230203', '20230206', '20230207', '20230208', '20230209', '20230213', '20230214']
npz_files = [f for f in os.listdir('.') if f.endswith('.npz') and any(sess in f for sess in sessions)]
all_states = []

for fname in npz_files:
    fname
    session_name = next((sess for sess in sessions if sess in fname), None)
    if session_name is None:
        continue
    data = np.load(fname)
    predicted_states = data['predicted_states']
    all_states.extend(predicted_states.ravel().tolist())

# --------------------------------------------------------------------
# State durations
# --------------------------------------------------------------------
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
n_states = len(np.unique(all_states))
transition_counts = np.zeros((n_states, n_states), dtype=int)
for a, b in zip(all_states[:-1], all_states[1:]):
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
        plt.text(j, i, f"{transition_prob[i,j]:.2f}", ha='center', va='center',
                 color='black' if transition_prob[i,j]<0.5 else 'white', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'state_transition_matrix.pdf'))
plt.close()

# --------------------------------------------------------------------
# Trial-level data extraction
# --------------------------------------------------------------------
all_trials = []
for fname in npz_files:
    session_name = next((sess for sess in sessions if sess in fname), None)
    session_log_dir = os.path.join(raw_data_dir, session_name)
    if not os.path.isdir(session_log_dir):
        continue
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
    predicted_states = np.load(fname)['predicted_states']
    rt = np.load(fname)['original_emissions'] if 'original_emissions' in np.load(fname).files else np.full_like(predicted_states, np.nan, dtype=float)

    for i,state in enumerate(predicted_states):
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
summary = trial_df.groupby('PredictedState').agg(
    correct_rate=('correct', 'mean'),
    incorrect_rate=('wrong', 'mean'),
    miss_rate=('misses', 'mean')
).reset_index()

# Filter only included states
summary = summary[summary['PredictedState'].isin(states_to_include)]
states = summary['PredictedState'].tolist()
matrix = np.vstack([
    summary['correct_rate'],
    summary['incorrect_rate'],
    summary['miss_rate']
])

# Pairwise significance testing between states
outcomes = ['correct_rate','incorrect_rate','miss_rate']
outcome_labels = ['Correct','Incorrect','Misses']

pairwise_significance = []
for s1, s2 in combinations(states_to_include, 2):
    df1 = trial_df[trial_df['PredictedState']==s1]
    df2 = trial_df[trial_df['PredictedState']==s2]
    for outcome, label in zip(outcomes, outcome_labels):
        col = label.lower() if label!='Incorrect' else 'wrong'
        count1 = df1[col].sum()
        n1 = len(df1)
        count2 = df2[col].sum()
        n2 = len(df2)
        table = [[count1, n1-count1],
                 [count2, n2-count2]]
        if min(table[0]+table[1]) < 5:
            stat, p = fisher_exact(table)
        else:
            stat, p, dof, expected = chi2_contingency(table)
        pairwise_significance.append({'State1': s1,'State2': s2,'Outcome': label,'p-value': p})
sig_df = pd.DataFrame(pairwise_significance)
sig_df['p-corrected'] = np.minimum(sig_df['p-value']*len(sig_df),1)
sig_df['significant'] = sig_df['p-corrected']<0.05

white_to_teal = LinearSegmentedColormap.from_list("white_to_teal", ["#FFFFFF", "#008080"])

# --- Settings for Illustrator-safe PDF ---
plt.rcParams['pdf.fonttype'] = 42  # Ensures fonts are saved as Type 42 (PostScript) for editability in Illustrator
plt.rcParams['ps.fonttype'] = 42   # Ensures fonts are saved as Type 42 (PostScript)
plt.rcParams['figure.facecolor'] = 'white'  # Set the figure background to white
plt.rcParams['savefig.facecolor'] = 'white'  # Ensure saved figure background is white
plt.rcParams['savefig.transparent'] = False  # Disable transparency

# --- Plotting ---
fig, ax = plt.subplots(figsize=(8,4))

# Use pcolormesh for full vector output (alternative to imshow)
c = ax.pcolormesh(
    np.arange(matrix.shape[1]+1)-0.5,
    np.arange(matrix.shape[0]+1)-0.5,
    matrix,
    cmap=white_to_teal,
    vmin=0,
    vmax=1,
    shading='flat'  # Flat shading ensures no anti-aliasing issues in the vector output
)
ax.set_xticks(np.arange(len(states)))
ax.set_xticklabels([f'State {int(s)}' for s in states])
ax.set_yticks([0,1,2])
ax.set_yticklabels(['Correct','Incorrect','Misses'])

for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        ax.text(j, i, f"{matrix[i,j]:.2f}", ha='center', va='center',
                color='black' if matrix[i,j]<0.5 else 'white', fontsize=11)

# --- Add significance stars directly on plot ---
for _, row in sig_df[sig_df['significant']].iterrows():
    outcome_idx = outcome_labels.index(row['Outcome'])
    x1 = states.index(row['State1'])
    x2 = states.index(row['State2'])
    y = outcome_idx
    mid_x = (x1+x2)/2
    ax.text(mid_x, y-0.25, '*', ha='center', va='center', color='black', fontsize=14)

plt.colorbar(c, ax=ax, label='Proportion')
plt.title('Trial Outcome Proportions by Predicted State')
plt.tight_layout()

output_file = os.path.join(output_dir, 'trial_outcome_vs_states_significance.pdf')
plt.savefig(output_file, dpi=600, transparent=False, facecolor='white')
plt.close()

# --------------------------------------------------------------------
# Reaction Time analysis (selected states)
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
state_colors = {0: (0.55, 0.0, 0.55), 1: (0.0, 0.39, 0.39), 2: (0.8, 0.33, 0.0)}
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
