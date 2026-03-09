"""
Plot histograms of trial lengths (from neural data) and reaction times
(from emissions.npy) across sessions.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import syncopy as spy

# -----------------------------
# Config
# -----------------------------
lfp_data_dir = '/cs/projects/MWzeronoise/Analysis/4Shivangi/Datasets/neural_data/stimAalign_cut/clean_full_length'
processed_dir = '/cs/projects/MWzeronoise/Analysis/4Shivangi/Datasets/states_analysis/processed'
output_dir = '/cs/projects/MWzeronoise/Analysis/4Shivangi/plots/preprocessing'
os.makedirs(output_dir, exist_ok=True)

sessions = {
    '20230203': 'Cosmos_20230203_LeafForaging_001',
    '20230208': 'Cosmos_20230208_LeafForaging_001',
    '20230209': 'Cosmos_20230209_LeafForaging_001',
    '20230213': 'Cosmos_20230213_LeafForaging_002',
    '20230214': 'Cosmos_20230214_LeafForaging_001',
}

# ============================================================
# 1. Trial length histograms (from neural data .time property)
# ============================================================
all_trial_lengths = []
session_trial_lengths = {}

for date in sessions:
    lfp_path = os.path.join(lfp_data_dir, date, 'Cleaned_lfp_FT.spy')
    if not os.path.exists(lfp_path):
        print(f"Skipping {date}: LFP data not found")
        continue

    datalfp = spy.load(lfp_path)
    # Each element of data.time is a time vector for one trial
    lengths = np.array([len(t) for t in datalfp.time])
    session_trial_lengths[date] = lengths
    all_trial_lengths.append(lengths)
    srate = datalfp.samplerate
    print(f"Session {date}: {len(lengths)} trials, "
          f"mean length={lengths.mean():.0f} samples ({lengths.mean()/srate:.3f}s), "
          f"std={lengths.std():.0f} samples")

all_trial_lengths = np.concatenate(all_trial_lengths)

# Plot trial length histograms
n_sessions = len(session_trial_lengths)
fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=True)
axes = axes.flatten()

for i, (date, lengths) in enumerate(session_trial_lengths.items()):
    ax = axes[i]
    # Convert to seconds using samplerate
    lengths_s = lengths / srate
    ax.hist(lengths_s, bins=50, range=(0, 20), color='steelblue', edgecolor='white', alpha=0.8)
    ax.set_xlim(0, 20)
    ax.axvline(np.median(lengths_s), color='red', ls='--', lw=1.5,
               label=f'median={np.median(lengths_s):.3f}s')
    ax.axvline(lengths_s.mean(), color='orange', ls='--', lw=1.5,
               label=f'mean={lengths_s.mean():.3f}s')
    ax.set_title(f'Session {date}')
    ax.set_xlabel('Trial Length (s)')
    ax.set_ylabel('Count')
    ax.legend(fontsize=8)

# Overall in last subplot
ax = axes[-1]
all_lengths_s = all_trial_lengths / srate
ax.hist(all_lengths_s, bins=60, range=(0, 5), color='gray', edgecolor='white', alpha=0.8)
ax.set_xlim(0, 5)
ax.axvline(np.median(all_lengths_s), color='red', ls='--', lw=1.5,
           label=f'median={np.median(all_lengths_s):.3f}s')
ax.axvline(all_lengths_s.mean(), color='orange', ls='--', lw=1.5,
           label=f'mean={all_lengths_s.mean():.3f}s')
ax.set_title('All Sessions')
ax.set_xlabel('Trial Length (s)')
ax.legend(fontsize=8)

fig.suptitle('Trial Length Distributions', fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
fname = os.path.join(output_dir, 'trial_length_histogram_cut5.pdf')
fig.savefig(fname)
plt.close(fig)
print(f"\nSaved: {fname}")

# ============================================================
# 2. Reaction time histograms (from emissions.npy)
# ============================================================
all_rts = []
session_rts = {}

for date, folder in sessions.items():
    emissions_path = os.path.join(processed_dir, folder, 'emissions.npy')
    if not os.path.exists(emissions_path):
        print(f"Skipping {date}: emissions.npy not found")
        continue
    rt = np.load(emissions_path).flatten()
    session_rts[date] = rt
    all_rts.append(rt)
    print(f"Session {date}: {len(rt)} trials, "
          f"mean RT={rt.mean():.3f}s, median={np.median(rt):.3f}s, "
          f"std={rt.std():.3f}s")

all_rts = np.concatenate(all_rts)
print(f"\nAll sessions: {len(all_rts)} trials, "
      f"mean RT={all_rts.mean():.3f}s, median={np.median(all_rts):.3f}s")

# Plot RT histograms
fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=True)
axes = axes.flatten()

for i, (date, rt) in enumerate(session_rts.items()):
    ax = axes[i]
    ax.hist(rt, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(np.median(rt), color='red', ls='--', lw=1.5,
               label=f'median={np.median(rt):.3f}s')
    ax.axvline(rt.mean(), color='orange', ls='--', lw=1.5,
               label=f'mean={rt.mean():.3f}s')
    ax.set_title(f'Session {date}')
    ax.set_xlabel('Reaction Time (s)')
    ax.set_ylabel('Count')
    ax.legend(fontsize=8)

ax = axes[-1]
ax.hist(all_rts, bins=60, color='gray', edgecolor='white', alpha=0.8)
ax.axvline(np.median(all_rts), color='red', ls='--', lw=1.5,
           label=f'median={np.median(all_rts):.3f}s')
ax.axvline(all_rts.mean(), color='orange', ls='--', lw=1.5,
           label=f'mean={all_rts.mean():.3f}s')
ax.set_title('All Sessions')
ax.set_xlabel('Reaction Time (s)')
ax.legend(fontsize=8)

fig.suptitle('Reaction Time Distributions', fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
fname = os.path.join(output_dir, 'RT_histograms.pdf')
fig.savefig(fname)
plt.close(fig)
print(f"\nSaved: {fname}")
