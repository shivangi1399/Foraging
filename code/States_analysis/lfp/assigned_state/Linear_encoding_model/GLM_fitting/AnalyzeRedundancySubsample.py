"""
=============================================================================================
Summarise the redundancy diagnostic (required step 3b).
=============================================================================================
Pipeline position: step 3b (see README.md). For every session/channel, reads the .jsonl that
RedundancySubsample.py wrote into that channel's results/ folder, then per channel: plots how
many columns get flagged redundant vs downscale_factor and how stable the redundant set is across
subsamples (Jaccard overlap). Saves a PDF into the plots tree and prints the Jaccard table.
Run after the RedundancySubsample sweep (step 3a) finishes.

Light / login-node script (no acme): `python AnalyzeRedundancySubsample.py`. Set DOWNSCALE_METHOD
below to match what RedundancySubsample.py wrote ('random' by default).
"""

import os
import re
import glob
from pathlib import Path
from itertools import combinations
from collections import Counter

import pandas as pd
import matplotlib
matplotlib.use("Agg")            
import matplotlib.pyplot as plt

results_dir = '/cs/projects/MWzeronoise/Analysis/4Shivangi/Results/states_analysis/states_lfp/all_trials/full_length/GLM'

# Figures are written here, mirroring the results tree (full_length/GLM/<session>/
# channel<ch>_regressors/results/) but rooted in the plots dir instead of Results.
plots_dir = '/cs/projects/MWzeronoise/Analysis/4Shivangi/plots/states_lfp/all_trials/full_length/GLM'

# Which session/channel pairs to summarise. None -> auto-discover from the results tree.
SESSIONS = None
CHANNELS = None

# Which sweep to read; must match RedundancySubsample.py's DOWNSCALE_METHOD.
DOWNSCALE_METHOD = 'random'


def discover_channels(session):
    chans = []
    for p in glob.glob(str(Path(results_dir) / session / 'channel*_regressors')):
        m = re.search(r'channel(\d+)_regressors$', os.path.basename(p))
        if m:
            chans.append(int(m.group(1)))
    return sorted(chans)


def discover_sessions():
    out = []
    for p in sorted(glob.glob(str(Path(results_dir) / '*'))):
        if os.path.isdir(p) and discover_channels(os.path.basename(p)):
            out.append(os.path.basename(p))
    return out


def jaccard(a, b):
    a, b = set(a), set(b)
    if not (a or b):
        return 1.0
    return len(a & b) / len(a | b)


def analyze_channel(session, channel):
    designMatID = f'{session}_channel{channel}'
    SAVE_PATH = Path(results_dir) / session / f'channel{channel}_regressors' / 'results'
    file_path = SAVE_PATH / f'{designMatID}_redundancy_{DOWNSCALE_METHOD}.jsonl'
    if not file_path.exists():
        print(f'skip {designMatID}: no {file_path.name} (run RedundancySubsample.py first)')
        return

    df = pd.read_json(file_path, lines=True)
    print(f'\n===== {designMatID} ({len(df)} subsamples) =====')

    # how many columns flagged redundant, by downscale factor
    rej_idx_lengths = df['rejIdx'].apply(len)
    factor_order = sorted(df['downscale_factor'].unique())
    groups = [rej_idx_lengths[df['downscale_factor'] == f].values for f in factor_order]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(groups, labels=factor_order, patch_artist=True)
    for x_pos, lengths in enumerate(groups, start=1):
        ax.scatter([x_pos] * len(lengths), lengths, color='black', alpha=0.7, s=20)
    ax.set_xlabel('downscale_factor')
    ax.set_ylabel('length of rejIdx')
    ax.set_title(f'{designMatID}: rejIdx length by downscale_factor')
    plt.tight_layout()
    fig_dir = Path(plots_dir) / session / f'channel{channel}_regressors' / 'results'
    fig_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = fig_dir / f'{designMatID}_redundancy_{DOWNSCALE_METHOD}.pdf'
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f'  saved plot -> {pdf_path}')

    # which regressor GROUPS get flagged redundant, and how consistently across subsamples.
    # NOTE: names come from RedundancySubsample.py's `redundant_regs`; trust the family
    # (difficulty / movement / state / correctness), not the exact level -- which specific
    # level of a collinear set QR picks is arbitrary.
    n = len(df)
    overall = Counter()
    for regs in df['redundant_regs']:
        overall.update(set(regs))
    if overall:
        print(f'  regressor groups flagged redundant (out of {n} subsamples):')
        for reg, k in overall.most_common():
            print(f'    {reg:24s} {k:3d}/{n}  ({100 * k / n:5.1f}%)')
        # per-factor breakdown (the redundant set that survives to the biggest factor is the
        # most trustworthy -- fewest rows, so only genuine dependence still shows up)
        print('  by downscale_factor:')
        for f in sorted(df['downscale_factor'].unique()):
            sub = df[df['downscale_factor'] == f]
            c = Counter()
            for regs in sub['redundant_regs']:
                c.update(set(regs))
            flagged = ', '.join(f'{r}({k}/{len(sub)})' for r, k in c.most_common()) or 'none'
            print(f'    factor {f:>3}: {flagged}')
    else:
        print('  no regressor groups flagged redundant in any subsample')

    # stability of the redundant set across subsamples (Jaccard overlap)
    overlap = []
    for ds, group in df.groupby('downscale_factor'):
        if len(group) < 2:
            continue
        for (_, row1), (_, row2) in combinations(group.iterrows(), 2):
            overlap.append({
                'downscale_factor': ds,
                'rejIdx_jaccard': jaccard(row1['rejIdx'], row2['rejIdx']),
                'regs_jaccard': jaccard(row1['redundant_regs'], row2['redundant_regs']),
            })
    if overlap:
        summary = pd.DataFrame(overlap).groupby('downscale_factor')[
            ['rejIdx_jaccard', 'regs_jaccard']].mean()
        print(summary)


if __name__ == '__main__':
    sessions = SESSIONS if SESSIONS is not None else discover_sessions()
    if not sessions:
        raise SystemExit(f'No sessions with channel*_regressors found under {results_dir}')
    for session in sessions:
        channels = CHANNELS if CHANNELS is not None else discover_channels(session)
        for channel in channels:
            analyze_channel(session, channel)
