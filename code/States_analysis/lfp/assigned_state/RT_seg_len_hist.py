"""
Plot RT-centered segment length histograms (one subplot per state).

Loads LFP and state data, extracts RT-centered segments, and plots the
distribution of segment lengths across all sessions per state.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import syncopy as spy

# -----------------------------
# Paths & Config
# -----------------------------
lfp_data_dir = '/cs/projects/MWzeronoise/Analysis/4Shivangi/Datasets/neural_data/stimAalign_cut/clean_full_length'
trial_info_dir = '/cs/projects/MWzeronoise/Analysis/4Shivangi/Datasets/neural_data/stimAalign_cut/full_length'
states_data_dir = '/cs/projects/MWzeronoise/Analysis/4Shivangi/Datasets/states_analysis'
processed_dir = '/cs/projects/MWzeronoise/Analysis/4Shivangi/Datasets/states_analysis/processed'
output_dir = '/cs/projects/MWzeronoise/Analysis/4Shivangi/plots/states_lfp/RT_aligned'
os.makedirs(output_dir, exist_ok=True)

sessions = ['20230203', '20230208', '20230209', '20230213', '20230214']
session_folders = {
    '20230203': 'Cosmos_20230203_LeafForaging_001',
    '20230208': 'Cosmos_20230208_LeafForaging_001',
    '20230209': 'Cosmos_20230209_LeafForaging_001',
    '20230213': 'Cosmos_20230213_LeafForaging_002',
    '20230214': 'Cosmos_20230214_LeafForaging_001',
}

N_STATES_TO_USE = 4
fs = 1000       # sampling rate in Hz
pre_rt = 0.45   # seconds before RT
post_rt = 0.45  # seconds after RT

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


def ensure_trialindex_in_trialdefinition(datalfp):
    if datalfp.trialdefinition.shape[1] < 4:
        nTrials = datalfp.trialdefinition.shape[0]
        datalfp.trialdefinition = np.hstack(
            (datalfp.trialdefinition, np.arange(nTrials).reshape(-1, 1))
        )


# -----------------------------
# Collect segment lengths
# -----------------------------
all_seg_lengths = {}  # state -> list of segment lengths
all_pre_samples = {}  # state -> list of samples available before RT
all_post_samples = {}  # state -> list of samples available after RT

for session_name in sessions:
    print(f"\n=== Processing session {session_name} ===")
    lfp_path = os.path.join(lfp_data_dir, session_name, 'Cleaned_lfp_FT.spy')
    trial_info_path = os.path.join(trial_info_dir, session_name, 'Trial_Info.pkl')
    emissions_path = os.path.join(processed_dir, session_folders[session_name], 'emissions.npy')

    if not os.path.exists(lfp_path) or not os.path.exists(trial_info_path):
        continue
    if not os.path.exists(emissions_path):
        print(f"  emissions.npy not found for {session_name}, skipping")
        continue

    # load RT data
    rt_values = np.load(emissions_path).flatten()

    # state info
    predicted_states = session_to_probs[session_name]
    trial_info_df = pd.read_pickle(trial_info_path)
    trial_info_df.iloc[:, 0] = (trial_info_df.iloc[:, 0] - 1000).astype('Int64')
    stim_df = pd.DataFrame({
        'TrialIndex': np.arange(len(predicted_states)),
        'States': predicted_states,
        'RT': rt_values[:len(predicted_states)]
    })
    combined_df = pd.merge(trial_info_df, stim_df, left_on='Trial_Number', right_on='TrialIndex', how='inner')

    # load LFP data
    datalfp = spy.load(lfp_path)
    ensure_trialindex_in_trialdefinition(datalfp)

    lfp_trial_indices = datalfp.trialdefinition[:, 3].astype(int)
    states_trial_info_filt = combined_df[combined_df['TrialIndex'].isin(lfp_trial_indices)]
    unique_states = np.sort(np.unique(states_trial_info_filt['States'].to_numpy()))[:N_STATES_TO_USE]

    for state_value in unique_states:
        state_trials = states_trial_info_filt[states_trial_info_filt['States'] == state_value]

        seg_lengths = []
        pre_samps = []
        post_samps = []
        for _, row in state_trials.iterrows():
            trial_idx = row['TrialIndex']
            rt = row['RT']

            lfp_trial_pos = np.where(lfp_trial_indices == trial_idx)[0]
            if len(lfp_trial_pos) == 0:
                continue
            lfp_trial_pos = lfp_trial_pos[0]

            trial_data = datalfp.trials[lfp_trial_pos]
            trial_time = datalfp.time[lfp_trial_pos]

            if np.all(np.isnan(trial_data)):
                continue

            t_start = rt - pre_rt
            t_end = rt + post_rt
            time_mask = (trial_time >= t_start) & (trial_time <= t_end)

            n_samples = np.sum(time_mask)
            if n_samples < 10:
                continue
            seg_lengths.append(n_samples)

            # samples available before and after RT (within trial bounds)
            n_pre = np.sum(trial_time < rt)       # how much data exists before RT
            n_post = np.sum(trial_time >= rt)      # how much data exists at/after RT
            pre_samps.append(n_pre)
            post_samps.append(n_post)

        if seg_lengths:
            if state_value not in all_seg_lengths:
                all_seg_lengths[state_value] = []
                all_pre_samples[state_value] = []
                all_post_samples[state_value] = []
            all_seg_lengths[state_value].extend(seg_lengths)
            all_pre_samples[state_value].extend(pre_samps)
            all_post_samples[state_value].extend(post_samps)

        print(f"  State {state_value}: {len(seg_lengths)} segments collected")

# -----------------------------
# Plot histogram
# -----------------------------
n_states_hist = len(all_seg_lengths)
if n_states_hist == 0:
    print("No segment lengths collected — nothing to plot.")
else:
    fig_hist, axes_hist = plt.subplots(1, n_states_hist, figsize=(5 * n_states_hist, 4), sharey=True)
    if n_states_hist == 1:
        axes_hist = [axes_hist]

    for idx, state_val in enumerate(sorted(all_seg_lengths.keys())):
        ax = axes_hist[idx]
        lengths = np.array(all_seg_lengths[state_val]) / fs  # convert to seconds
        p10 = np.percentile(lengths, 10)
        p15 = np.percentile(lengths, 15)
        p20 = np.percentile(lengths, 20)
        mean_len = lengths.mean()
        median_len = np.median(lengths)
        mode_bin = int(np.argmax(np.bincount((lengths * fs).astype(int)))) / fs
        mode_len = mode_bin

        ax.hist(lengths, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
        ax.axvline(mean_len, color='blue', ls='-', lw=1.5, label=f'mean={mean_len:.3f}s')
        ax.axvline(median_len, color='purple', ls='-', lw=1.5, label=f'median={median_len:.3f}s')
        ax.axvline(mode_len, color='cyan', ls='-', lw=1.5, label=f'mode={mode_len:.3f}s')
        n_total = len(lengths)
        n_p10 = int(np.sum(lengths >= p10))
        n_p15 = int(np.sum(lengths >= p15))
        n_p20 = int(np.sum(lengths >= p20))
        ax.axvline(p10, color='red', ls='--', lw=1.5, label=f'p10={p10:.3f}s ({n_p10}/{n_total})')
        ax.axvline(p15, color='orange', ls='--', lw=1.5, label=f'p15={p15:.3f}s ({n_p15}/{n_total})')
        ax.axvline(p20, color='green', ls='--', lw=1.5, label=f'p20={p20:.3f}s ({n_p20}/{n_total})')
        ax.set_xlabel('Segment length (s)')
        if idx == 0:
            ax.set_ylabel('Count')
        ax.set_title(f'State {state_val} (n={len(lengths)})')
        ax.legend(fontsize=7)

    plt.suptitle('RT-centered segment length distribution (all sessions pooled)', fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    hist_fname = os.path.join(output_dir, 'seg_length_hist_all_sess.pdf')
    fig_hist.savefig(hist_fname)
    plt.close(fig_hist)
    print(f"\nHistogram saved: {hist_fname}")

    # ----- Pre/Post RT availability plot -----
    fig_pp, axes_pp = plt.subplots(2, n_states_hist, figsize=(5 * n_states_hist, 7), sharey='row')
    if n_states_hist == 1:
        axes_pp = axes_pp.reshape(-1, 1)

    for idx, state_val in enumerate(sorted(all_pre_samples.keys())):
        pre = np.array(all_pre_samples[state_val]) / fs  # convert to seconds
        post = np.array(all_post_samples[state_val]) / fs

        for row, (data, label, color) in enumerate([
            (pre, 'Pre-RT (s before RT)', 'coral'),
            (post, 'Post-RT (s after RT)', 'teal'),
        ]):
            ax = axes_pp[row, idx]
            ax.hist(data, bins=50, color=color, edgecolor='white', alpha=0.8)
            p10 = np.percentile(data, 10)
            ax.axvline(data.mean(), color='blue', ls='-', lw=1.5, label=f'mean={data.mean():.3f}s')
            ax.axvline(np.median(data), color='purple', ls='-', lw=1.5, label=f'median={np.median(data):.3f}s')
            ax.axvline(p10, color='red', ls='--', lw=1.5, label=f'p10={p10:.3f}s')
            ax.set_xlabel(label)
            if idx == 0:
                ax.set_ylabel('Count')
            ax.set_title(f'State {state_val} (n={len(data)})')
            ax.legend(fontsize=7)

    plt.suptitle('Data available on each side of RT (all sessions pooled)', fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    pp_fname = os.path.join(output_dir, 'pre_post_rt_hist_all_sess.pdf')
    fig_pp.savefig(pp_fname)
    plt.close(fig_pp)
    print(f"Pre/Post RT histogram saved: {pp_fname}")
