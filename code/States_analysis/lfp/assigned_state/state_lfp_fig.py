"""
Generates combined plots for:
1. ERP (timelock) and residual spectra with significance masks.
2. Trial-wise coherence results and array wise comparison.

"""

# -----------------------------
# Imports
# -----------------------------
import os
import numpy as np
import matplotlib.pyplot as plt
import itertools
from matplotlib.colors import LinearSegmentedColormap

# -----------------------------
# User Config
# -----------------------------
results_data_dir = '/mnt/cs/projects/MWzeronoise/Analysis/4Shivangi/Results/states_analysis/states_lfp/200_600/all_trials'
output_dir = '/cs/projects/MWzeronoise/Analysis/4Shivangi/plots/states_lfp/all_trials/200_600/erp_spectra/summary_plots'
os.makedirs(output_dir, exist_ok=True)

plot_types = ['timelock', 'residual']
states = [0, 1, 2, 3]
arrays = [1, 2, 3, 4, 5, 6]
state_colors = {
    0: (0.55, 0.0, 0.55),   # purple
    1: (0.0, 0.39, 0.39),   # teal
    2: (0.8, 0.33, 0.0),     # orange
    3: (0.25, 0.35, 0.55)   # slate blue
}
sig_color = '#8dd3c7'  # teal for significance mask
teal_cmap = LinearSegmentedColormap.from_list('teal_map', ['white', '#1f9e89'])


def load_permdata(plot_type, s1, s2, array_index):
    """Load permutation test results for a given array and state pair."""
    fname = f"permdata_{plot_type}_pair{s1}_{s2}_ARRAY_array{array_index}.npz"
    fpath = os.path.join(results_data_dir, fname)
    if not os.path.exists(fpath):
        return None
    return np.load(fpath)

# =============================================================
# Real ERP and Residual Spectra
# =============================================================
pairs = list(itertools.combinations(states, 2))

for plot_type in plot_types:
    ylabel = 'Amplitude' if plot_type == 'timelock' else 'Residual Power'
    n_rows, n_cols = len(arrays), len(pairs)

    # -----------------------------
    # Figure 1: Real data + significance masks
    # -----------------------------
    fig_real, axes_real = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 3*n_rows), sharex='col', sharey='row')
    if n_rows == 1: axes_real = np.expand_dims(axes_real, 0)
    if n_cols == 1: axes_real = np.expand_dims(axes_real, 1)

    summary_mask = []

    for i_array, array_index in enumerate(arrays):
        summary_mask_array = []
        for j_pair, (s1, s2) in enumerate(pairs):
            ax = axes_real[i_array, j_pair]
            perm_array = load_permdata(plot_type, s1, s2, array_index)

            if perm_array is None:
                ax.axis('off')
                summary_mask_array.append(None)
                continue

            # Extract data
            x_axis = perm_array['x_axis']
            data1, data2 = perm_array['mean1'], perm_array['mean2']
            sig_mask = perm_array['sig']

            # Plot means
            ax.plot(x_axis, data1, color=state_colors[s1], label=f"State {s1}")
            ax.plot(x_axis, data2, color=state_colors[s2], label=f"State {s2}")

            # Add significance shading
            ax.fill_between(x_axis, ax.get_ylim()[0], ax.get_ylim()[1], where=sig_mask,
                            color=sig_color, alpha=0.4)

            # Labels and titles
            if i_array == 0: ax.set_title(f"{s1} vs {s2}", fontsize=10)
            if j_pair == 0: ax.set_ylabel(f"Array {array_index}\n{ylabel}")
            if i_array == n_rows - 1:
                xlabel = 'Time (s)' if plot_type == 'timelock' else 'Frequency (Hz)'
                ax.set_xlabel(xlabel)
            ax.legend(fontsize=6)

            summary_mask_array.append(sig_mask)
        summary_mask.append(summary_mask_array)

    plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.08, wspace=0.25, hspace=0.25)
    plt.suptitle(f"{plot_type} - Mean across all sessions", fontsize=14)
    plt.savefig(os.path.join(output_dir, f"{plot_type}_all_arrays_pairs.pdf"))
    plt.close()

    # -----------------------------
    # Figure 2: Combined summary heatmap
    # -----------------------------
    n_arrays, n_pairs, n_time = len(arrays), len(pairs), len(x_axis)
    summary_array = np.zeros((n_arrays, n_pairs, n_time), dtype=int)

    for i_array in range(n_arrays):
        for j_pair in range(n_pairs):
            mask = summary_mask[i_array][j_pair]
            if mask is not None and mask.shape[0] == n_time:
                summary_array[i_array, j_pair, :] = mask.astype(int)

    fig_sum, axes_sum = plt.subplots(1, n_pairs, figsize=(6*n_pairs, 4), sharey=True)
    if n_pairs == 1: axes_sum = [axes_sum]

    for j_pair, (s1, s2) in enumerate(pairs):
        ax = axes_sum[j_pair]
        im = ax.imshow(summary_array[:, j_pair, :], cmap=teal_cmap, aspect='auto', interpolation='none',
                       extent=[x_axis[0], x_axis[-1], 0.5, n_arrays + 0.5])
        ax.set_title(f"{s1} vs {s2}")
        ax.set_xlabel('Time (s)' if plot_type == 'timelock' else 'Frequency (Hz)')
        if j_pair == 0:
            ax.set_ylabel('Arrays')
            ax.set_yticks(range(1, n_arrays + 1))
            ax.set_yticklabels([str(a) for a in arrays])

    # Colorbar
    plt.subplots_adjust(left=0.05, right=0.88, top=0.88, bottom=0.12, wspace=0.3)
    cbar_ax = fig_sum.add_axes([0.90, 0.12, 0.02, 0.76])
    cbar = fig_sum.colorbar(im, cax=cbar_ax)
    cbar.set_label('Significant (1=pairwise)')

    plt.suptitle(f"{plot_type} - Pairwise significance across arrays and state pairs", fontsize=14)
    plt.savefig(os.path.join(output_dir, f"{plot_type}_pairwise_summary_all_arrays.pdf"))
    plt.close()


# -----------------------------
# Figure 3: Merged Array for ERP and Spectra Analysis
# -----------------------------

def load_permdata_merged(plot_type, s1, s2, array_label):
    """Load permutation data for merged array sets."""
    fname = f"cb_permdata_{plot_type}_pair{s1}_{s2}_{array_label.replace('-', '')}.npz"
    fpath = os.path.join(results_data_dir, fname)
    if not os.path.exists(fpath):
        return None
    return np.load(fpath)

array_labels = ['Array 1-3', 'Array 4', 'Array 5', 'Array 6']
pairs = list(itertools.combinations(states, 2))

for plot_type in plot_types:
    ylabel = 'Amplitude' if plot_type == 'timelock' else 'Residual Power'
    n_rows, n_cols = len(pairs), len(array_labels)  # <-- swapped rows and columns

    # Figure: Merged arrays
    fig_real, axes_real = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 3*n_rows), sharex='col', sharey='row')
    if n_rows == 1: axes_real = np.expand_dims(axes_real, 0)
    if n_cols == 1: axes_real = np.expand_dims(axes_real, 1)

    summary_mask = []

    for i_pair, (s1, s2) in enumerate(pairs):  # <-- outer loop over pairs (rows)
        summary_mask_row = []
        for j_array, array_label in enumerate(array_labels):  # <-- inner loop over arrays (columns)
            ax = axes_real[i_pair, j_array]
            perm_array = load_permdata_merged(plot_type, s1, s2, array_label)

            if perm_array is None:
                ax.axis('off')
                summary_mask_row.append(None)
                continue

            x_axis = perm_array['x_axis']
            data1, data2 = perm_array['mean1'], perm_array['mean2']
            sig_mask = perm_array['sig']

            ax.plot(x_axis, data1, color=state_colors[s1], label=f"State {s1}")
            ax.plot(x_axis, data2, color=state_colors[s2], label=f"State {s2}")
            ax.fill_between(x_axis, ax.get_ylim()[0], ax.get_ylim()[1], where=sig_mask,
                            color=sig_color, alpha=0.4)

            if i_pair == 0: ax.set_title(f"{array_label}", fontsize=10)  # <-- columns: arrays
            if j_array == 0: ax.set_ylabel(f"{s1} vs {s2}\n{ylabel}")   # <-- rows: state pairs
            if i_pair == n_rows - 1:
                xlabel = 'Time (s)' if plot_type == 'timelock' else 'Frequency (Hz)'
                ax.set_xlabel(xlabel)
            ax.legend(fontsize=6)

            summary_mask_row.append(sig_mask)
        summary_mask.append(summary_mask_row)

    plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.08, wspace=0.25, hspace=0.25)
    plt.suptitle(f"{plot_type} - Mean across all sessions (merged arrays)", fontsize=14)
    plt.savefig(os.path.join(output_dir, f"{plot_type}_merged_arrays_pairs.pdf"))
    plt.close()

print(f"All plots (real data + summary) saved under {output_dir}")

# =============================================================
# Coherence
# =============================================================

results_dir = '/mnt/cs/projects/MWzeronoise/Analysis/4Shivangi/Results/states_analysis/states_lfp/coherence_scipy_trialwise'
output_dir = '/cs/projects/MWzeronoise/Analysis/4Shivangi/plots/states_lfp/all_trials/200_600/coherence_scipy_trialwise/summary_plots'
os.makedirs(output_dir, exist_ok=True)

def load_permdata_trialwise_coh(s1, s2, arr_pair):
    """Load trial-wise coherence permutation test results."""
    fname = f"permdata_trialwise_coh_pair{s1}_{s2}_array{arr_pair[0]}-{arr_pair[1]}.npz"
    fpath = os.path.join(results_dir, fname)
    if not os.path.exists(fpath):
        return None
    return np.load(fpath)

# -----------------------------
# Figure 1: Trial-wise Coherence - All Array Pairs
# -----------------------------
pairs = list(itertools.combinations(states, 2))
unique_array_pairs = [(i, j) for i in arrays for j in arrays if j > i]

n_rows, n_cols = len(unique_array_pairs), len(pairs)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 2.5*n_rows), sharex='col', sharey='row')
if n_rows == 1: axes = np.expand_dims(axes, 0)
if n_cols == 1: axes = np.expand_dims(axes, 1)

summary_mask = []

for i_arr, arr_pair in enumerate(unique_array_pairs):
    summary_mask_array = []
    for j_pair, (s1, s2) in enumerate(pairs):
        ax = axes[i_arr, j_pair]
        perm_data = load_permdata_trialwise_coh(s1, s2, arr_pair)

        if perm_data is None:
            ax.axis('off')
            summary_mask_array.append(None)
            continue

        f_axis = perm_data['x_axis']
        mean1 = perm_data['mean1']
        mean2 = perm_data['mean2']
        sig_mask = perm_data['sig']

        # Plot means
        ax.plot(f_axis, mean1, color=state_colors[s1], label=f"State {s1}", linewidth=1.5)
        ax.plot(f_axis, mean2, color=state_colors[s2], label=f"State {s2}", linewidth=1.5)
        
        # Add significance shading
        ax.fill_between(f_axis, ax.get_ylim()[0], ax.get_ylim()[1], 
                        where=sig_mask, color=sig_color, alpha=0.3)

        # Labels and titles
        if i_arr == 0: 
            ax.set_title(f"{s1} vs {s2}", fontsize=10)
        if j_pair == 0: 
            ax.set_ylabel(f"Array {arr_pair[0]}-{arr_pair[1]}\nTrial-wise Coherence")
        if i_arr == n_rows - 1: 
            ax.set_xlabel("Frequency (Hz)")
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)
        
        summary_mask_array.append(sig_mask)
    summary_mask.append(summary_mask_array)

plt.subplots_adjust(left=0.05, right=0.95, top=0.93, bottom=0.08, wspace=0.25, hspace=0.25)
plt.suptitle("Trial-wise Coherence - Mean across all sessions", fontsize=14)
plt.savefig(os.path.join(output_dir, "trialwise_coherence_all_arraypairs.pdf"), dpi=300, bbox_inches='tight')
plt.close()

print(f"Trial-wise coherence plots saved to {output_dir}")

# -----------------------------
# Figure 2: Summary Heatmaps by State Pair
# -----------------------------
# Get frequency axis from first available file
f_axis = None
for arr_pair in unique_array_pairs:
    for s1, s2 in pairs:
        perm_data = load_permdata_trialwise_coh(s1, s2, arr_pair)
        if perm_data is not None:
            f_axis = perm_data['x_axis']
            break
    if f_axis is not None:
        break

if f_axis is not None:
    n_freq = len(f_axis)
    n_array_pairs = len(unique_array_pairs)
    n_pairs = len(pairs)
    
    # Create summary array: (n_array_pairs, n_state_pairs, n_freq)
    summary_array = np.zeros((n_array_pairs, n_pairs, n_freq), dtype=int)
    
    for i_arr, arr_pair in enumerate(unique_array_pairs):
        for j_pair, (s1, s2) in enumerate(pairs):
            mask = summary_mask[i_arr][j_pair]
            if mask is not None and len(mask) == n_freq:
                summary_array[i_arr, j_pair, :] = mask.astype(int)
    
    # Plot heatmap for each state pair
    fig, axes = plt.subplots(1, n_pairs, figsize=(6*n_pairs, 5), sharey=True)
    if n_pairs == 1: 
        axes = [axes]
    
    for j_pair, (s1, s2) in enumerate(pairs):
        ax = axes[j_pair]
        im = ax.imshow(summary_array[:, j_pair, :], cmap=teal_cmap, aspect='auto', 
                      interpolation='none',
                      extent=[f_axis[0], f_axis[-1], 0.5, n_array_pairs + 0.5])
        ax.set_title(f"State {s1} vs {s2}")
        ax.set_xlabel('Frequency (Hz)')
        
        if j_pair == 0:
            ax.set_ylabel('Array Pairs')
            ax.set_yticks(range(1, n_array_pairs + 1))
            ax.set_yticklabels([f"{arr[0]}-{arr[1]}" for arr in unique_array_pairs])
    
    # Colorbar
    plt.subplots_adjust(left=0.08, right=0.88, top=0.88, bottom=0.12, wspace=0.3)
    cbar_ax = fig.add_axes([0.90, 0.12, 0.02, 0.76])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Significant (1=Yes, 0=No)')
    
    plt.suptitle("Trial-wise Coherence - Significance Summary Across Array Pairs", fontsize=14)
    plt.savefig(os.path.join(output_dir, "trialwise_coherence_summary_heatmap.pdf"), 
                dpi=300, bbox_inches='tight')
    plt.close()

# -----------------------------
# Figure 3: State Pair Comparison 
# -----------------------------
for s1, s2 in pairs:
    fig, axes = plt.subplots(len(unique_array_pairs), 1, 
                            figsize=(10, 3*len(unique_array_pairs)), 
                            sharex=True)
    if len(unique_array_pairs) == 1:
        axes = [axes]
    
    for i_arr, arr_pair in enumerate(unique_array_pairs):
        ax = axes[i_arr]
        perm_data = load_permdata_trialwise_coh(s1, s2, arr_pair)
        
        if perm_data is None:
            ax.axis('off')
            continue
        
        f_axis = perm_data['x_axis']
        mean1 = perm_data['mean1']
        mean2 = perm_data['mean2']
        sig_mask = perm_data['sig']
        
        # Plot with shaded error regions if available
        ax.plot(f_axis, mean1, color=state_colors[s1], label=f"State {s1}", linewidth=2)
        ax.plot(f_axis, mean2, color=state_colors[s2], label=f"State {s2}", linewidth=2)
        
        # Highlight significant regions
        ax.fill_between(f_axis, ax.get_ylim()[0], ax.get_ylim()[1], 
                        where=sig_mask, color=sig_color, alpha=0.3, 
                        label='Significant')
        
        ax.set_ylabel(f"Array {arr_pair[0]}-{arr_pair[1]}\nCoherence")
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        if i_arr == len(unique_array_pairs) - 1:
            ax.set_xlabel("Frequency (Hz)")
    
    plt.suptitle(f"Trial-wise Coherence: State {s1} vs State {s2}", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"trialwise_coherence_comparison_states_{s1}_{s2}.pdf"), 
                dpi=300, bbox_inches='tight')
    plt.close()

# -----------------------------
# Figure 4: Frequency Band Summary
# -----------------------------
# Define frequency bands
freq_bands = {
    'Delta': (2, 4),
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Beta': (13, 30),
    'Low Gamma': (30, 50),
    'High Gamma': (50, 100)
}

# Calculate percentage significant in each band
band_summary = {}  # (state_pair, array_pair, band) -> % significant

for s1, s2 in pairs:
    for arr_pair in unique_array_pairs:
        perm_data = load_permdata_trialwise_coh(s1, s2, arr_pair)
        
        if perm_data is None:
            continue
            
        f_axis = perm_data['x_axis']
        sig_mask = perm_data['sig']
        
        for band_name, (f_low, f_high) in freq_bands.items():
            band_mask = (f_axis >= f_low) & (f_axis <= f_high)
            if np.sum(band_mask) > 0:
                pct_sig = 100 * np.sum(sig_mask[band_mask]) / np.sum(band_mask)
                band_summary[((s1, s2), arr_pair, band_name)] = pct_sig

# Plot band summary
fig, axes = plt.subplots(len(pairs), 1, figsize=(12, 4*len(pairs)))
if len(pairs) == 1:
    axes = [axes]

for idx, (s1, s2) in enumerate(pairs):
    ax = axes[idx]
    
    x_pos = np.arange(len(freq_bands))
    width = 0.8 / len(unique_array_pairs)
    
    for i, arr_pair in enumerate(unique_array_pairs):
        values = []
        for band_name in freq_bands.keys():
            key = ((s1, s2), arr_pair, band_name)
            values.append(band_summary.get(key, 0))
        
        offset = (i - len(unique_array_pairs)/2 + 0.5) * width
        ax.bar(x_pos + offset, values, width, 
               label=f"Array {arr_pair[0]}-{arr_pair[1]}", alpha=0.8)
    
    ax.set_ylabel('% Significant')
    ax.set_title(f"State {s1} vs State {s2}")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(freq_bands.keys(), rotation=45, ha='right')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 100])

plt.suptitle("Frequency Band Significance Summary", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "trialwise_coherence_band_summary.pdf"), 
            dpi=300, bbox_inches='tight')
plt.close()

print(f"\nAll summary plots saved to {output_dir}")