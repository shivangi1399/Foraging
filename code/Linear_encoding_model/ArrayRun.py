"""
=============================================================================================
Run the WHOLE GLM pipeline over all channels, one electrode ARRAY at a time.
=============================================================================================
This wrapper walks the arrays one at a time; for each array it runs the requested pipeline STEPS 
(each step = one acme SLURM submission over just that array's channels), then writes a per-array
contribution summary. A combined summary is written at the end.

Pipeline STEPS (full chain, in dependency order -- do NOT reorder 'regressors'/'neural'):
    'regressors'    -> GLM_input/Regressors.process_channel        (per-channel regressors + full_mask_reg.npz;
                                                                    needs a per-session bundle built first)
    'neural'        -> GLM_input/NeuralData.process_channel        (neural_data.npz; READS full_mask_reg.npz,
                                                                    so it must run AFTER 'regressors')
    'design'        -> GLM_fitting/DesignMatrix.process_channel    (design matrix + downsampled target)
    'fit'           -> GLM_fitting/FittingGLM.fit_channel          (ridge fit -> {..}_samples.pkl)
    'contributions' -> GLM_fitting/RegressorContributions.contributions_channel ({..}_contributions.npz)
Each step reads the previous step's outputs, so trim STEPS to what you still need (e.g. just
['contributions'] to rebuild summaries).

IMPORTANT: 'neural'/'regressors' CREATE the channel folders, so on a fresh run the channels cannot be
discovered from disk -- set ALL_CHANNELS to the full channel list below.

Outputs (under SUMMARY_DIR):
    array{N}_contributions.csv       one row per channel: full_R2 + dR2/trace_var per family
    all_arrays_contributions.csv     concatenated, with an `array` column
    all_arrays_dR2_by_family.csv     wide: rows = channels, cols = per-family dR2 (quick scan)

Run in the warping env:  python RunByArray.py
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd

from acme import ParallelMap

# Locate the GLM_fitting / GLM_input package dirs regardless of where this wrapper lives
_HERE = os.path.dirname(os.path.abspath(__file__))
def _add_pkg(sub):
    for cand in (os.path.join(_HERE, sub), os.path.join(_HERE, '..', sub)):
        if os.path.isdir(cand):
            sys.path.insert(0, os.path.abspath(cand))
            return
    raise ImportError(f'could not locate {sub}/ relative to {_HERE}')
_add_pkg('GLM_fitting')
_add_pkg('GLM_input')
import DesignMatrix as dm
import FittingGLM as fg
import RegressorContributions as rc
import NeuralData as nd
import Regressors as rg

# -------------------------
# Config
# -------------------------
N_ARRAYS = 6                      # matches erp_spectra_stats.py (np.array_split(channels, 6))
CHANNELS_PER_ARRAY = 32           # probe layout: 6 arrays x 32 channels = 192 channels total
STEPS = ['regressors', 'neural', 'design', 'fit', 'contributions']   # dependency order; trim to what you need
ARRAYS_TO_RUN = [1]               # None -> all 6; or e.g. [1, 2] (1-based) to do a subset
DRY_RUN = False                   # True -> just print the array->channel split and exit

# Set to None to instead discover from existing channel*_regressors folders (only valid AFTER
# 'neural'/'regressors' have run).
ALL_CHANNELS = list(range(N_ARRAYS * CHANNELS_PER_ARRAY))   # [0, 1, ..., 191]
# Sessions to run. None -> use NeuralData's session config (works before any folders exist).
SESSIONS = ['20230214']

results_dir = rc.results_dir      # same GLM tree for every step
SUMMARY_DIR = os.path.join(results_dir, '_contribution_summaries')


# -------------------------
# Per-session setup for steps that need it (Regressors builds a session bundle first)
# -------------------------
def _regressors_prep(session):
    """Build the per-session regressor bundle on the login node (as Regressors.py's __main__ does)."""
    print(f"    [regressors] building session bundle for {session} ...")
    bundle = rg.compute_session(session)
    session_out = os.path.join(rg.results_dir, session)
    os.makedirs(session_out, exist_ok=True)
    bundle_path = os.path.join(session_out, '_session_bundle.pkl')
    with open(bundle_path, 'wb') as f:
        pickle.dump(bundle, f)
    return {'bundle_path': bundle_path}


def _regressors_cleanup(session, ctx):
    try:
        os.remove(ctx['bundle_path'])
    except OSError:
        pass


# step name -> config. `extra_args(session, ctx)` returns the positional args ParallelMap broadcasts
# after (channels, session); `prep`/`cleanup` are optional per-session hooks (run once, reused across
# arrays, cleaned up at the very end).
STEP_REGISTRY = {
    'neural': dict(
        worker=nd.process_channel, partition=nd.SLURM_PARTITION, mem=nd.MEM_PER_WORKER,
        max_workers=nd.MAX_WORKERS,
        extra_args=lambda session, ctx: (nd.lfp_data_dir, nd.log_file_dir, nd.results_dir),
    ),
    'regressors': dict(
        worker=rg.process_channel, partition=rg.SLURM_PARTITION, mem=rg.MEM_PER_WORKER,
        max_workers=rg.MAX_WORKERS,
        prep=_regressors_prep, cleanup=_regressors_cleanup,
        extra_args=lambda session, ctx: (ctx['bundle_path'], rg.rf_base_dir, rg.results_dir),
    ),
    'design': dict(
        worker=dm.process_channel, partition=dm.SLURM_PARTITION, mem=dm.MEM_PER_WORKER,
        max_workers=dm.MAX_WORKERS, extra_args=lambda session, ctx: (),
    ),
    'fit': dict(
        worker=fg.fit_channel, partition=fg.SLURM_PARTITION, mem=fg.MEM_PER_WORKER,
        max_workers=fg.MAX_WORKERS, extra_args=lambda session, ctx: (),
    ),
    'contributions': dict(
        worker=rc.contributions_channel, partition=rc.SLURM_PARTITION, mem=rc.MEM_PER_WORKER,
        max_workers=rc.MAX_WORKERS, extra_args=lambda session, ctx: (),
    ),
}

# per-session context for steps with a prep hook, built lazily and reused across arrays.
_session_ctx = {}


# -------------------------
# Array split (mirror erp_spectra_stats.py)
# -------------------------
def arrays_for_session(session):
    """List of N_ARRAYS channel-lists: the channel list split into contiguous groups."""
    channels = ALL_CHANNELS if ALL_CHANNELS is not None else rc.discover_channels(session)
    return [list(a) for a in np.array_split(np.asarray(channels), N_ARRAYS)]


# -------------------------
# Run one (step, session, channels) as a single acme submission
# -------------------------
def run_step(step, session, channels):
    if not channels:
        return
    cfg = STEP_REGISTRY[step]
    # per-session prep (once), reused across arrays
    key = (step, session)
    if key not in _session_ctx and 'prep' in cfg:
        _session_ctx[key] = cfg['prep'](session)
    ctx = _session_ctx.get(key, {})

    extra = cfg['extra_args'](session, ctx)
    n_workers = min(cfg['max_workers'], len(channels))
    print(f"    [{step}] session {session}: {len(channels)} channels -> {n_workers} workers on '{cfg['partition']}'")
    with ParallelMap(cfg['worker'], channels, session, *extra,
                     n_inputs=len(channels),
                     partition=cfg['partition'],
                     n_workers=n_workers,
                     mem_per_worker=cfg['mem'],
                     setup_timeout=600,            # busy cluster: wait up to 10 min for SLURM
                     write_worker_results=False,   # workers write their own npz/pkl
                     setup_interactive=False) as pmap:
        pmap.compute()


# -------------------------
# Build the contribution summary for one array (reads each channel's contributions.npz)
# -------------------------
def summarise_array(arr_idx, session_channels):
    """arr_idx is 1-based. session_channels: list of (session, channel). Returns a DataFrame."""
    rows = []
    for session, channel in session_channels:
        SAVE_PATH = os.path.join(results_dir, session, f'channel{channel}_regressors', 'results')
        designMatID = f'{session}_channel{channel}'
        f = os.path.join(SAVE_PATH, f'{designMatID}_contributions.npz')
        if not os.path.exists(f):
            print(f"    (no contributions.npz for {designMatID} -- skipped/failed)")
            continue
        c = np.load(f, allow_pickle=True)
        labels = [str(x) for x in c['group_labels']]
        row = {'array': arr_idx, 'session': session, 'channel': channel,
               'full_R2': float(c['full_R2'])}
        for lab, d, v in zip(labels, c['dR2'], c['trace_var']):
            row[f'dR2::{lab}'] = float(d)
            row[f'tracevar::{lab}'] = float(v)
        rows.append(row)
    return pd.DataFrame(rows)


# -------------------------
# Driver
# -------------------------
if __name__ == '__main__':
    os.makedirs(SUMMARY_DIR, exist_ok=True)
    sessions = SESSIONS if SESSIONS is not None else list(nd.sessions)

    # Build the per-session array split up front and show it.
    split = {s: arrays_for_session(s) for s in sessions}
    print(f"Sessions: {sessions}   STEPS: {STEPS}")
    expected_total = N_ARRAYS * CHANNELS_PER_ARRAY
    for s in sessions:
        total = sum(len(a) for a in split[s])
        sizes = [len(a) for a in split[s]]
        print(f"  {s}: {total} channels, array sizes {sizes}")
        for i, ch in enumerate(split[s], start=1):
            print(f"    array {i}: {len(ch)} channels {ch}")
        # consistency check: with the master list we expect 6 arrays x 32 = 192 channels
        if ALL_CHANNELS is not None and (total != expected_total or set(sizes) != {CHANNELS_PER_ARRAY}):
            raise SystemExit(
                f"Inconsistent split for {s}: got {total} channels in sizes {sizes}, expected "
                f"{expected_total} in {N_ARRAYS} arrays of {CHANNELS_PER_ARRAY}. Check N_ARRAYS / "
                f"CHANNELS_PER_ARRAY / ALL_CHANNELS.")

    # guard: input steps need an explicit channel list (folders don't exist yet on a fresh run)
    input_steps = {'neural', 'regressors'} & set(STEPS)
    if input_steps and all(sum(len(a) for a in split[s]) == 0 for s in sessions):
        raise SystemExit(
            f"STEPS include {sorted(input_steps)} but no channels were found. On a fresh run set "
            f"ALL_CHANNELS to the full channel list (folders don't exist yet to discover from).")

    if DRY_RUN:
        raise SystemExit("DRY_RUN: printed the array split only.")

    arr_indices = ARRAYS_TO_RUN if ARRAYS_TO_RUN is not None else list(range(1, N_ARRAYS + 1))
    all_summaries = []
    try:
        for arr_idx in arr_indices:
            print(f"\n================= ARRAY {arr_idx}/{N_ARRAYS} =================")
            session_channels = [(s, ch) for s in sessions for ch in split[s][arr_idx - 1]]
            if not session_channels:
                print("  no channels in this array, skipping")
                continue

            # run each requested step for this array's channels, one step at a time, per session
            for step in STEPS:
                print(f"  -- step: {step}")
                for s in sessions:
                    run_step(step, s, split[s][arr_idx - 1])

            # summarise this array's contributions (if that step is part of the run / already on disk)
            df = summarise_array(arr_idx, session_channels)
            if df.empty:
                print(f"  ARRAY {arr_idx}: no contribution files to summarise")
                continue
            out = os.path.join(SUMMARY_DIR, f'array{arr_idx}_contributions.csv')
            df.to_csv(out, index=False)
            all_summaries.append(df)
            dR2_cols = [c for c in df.columns if c.startswith('dR2::')]
            mean_dR2 = df[dR2_cols].mean().sort_values(ascending=False)
            print(f"  ARRAY {arr_idx}: {len(df)} channels summarised -> {out}")
            print(f"    mean full_R2 = {df['full_R2'].mean():.4f}")
            print(f"    top families by mean unique dR2: "
                  + ", ".join(f"{c.split('::')[1]}={mean_dR2[c]:+.4f}" for c in mean_dR2.index[:3]))
    finally:
        # clean up any per-session prep artifacts (e.g. Regressors bundles)
        for (step, session), ctx in _session_ctx.items():
            cleanup = STEP_REGISTRY[step].get('cleanup')
            if cleanup:
                cleanup(session, ctx)

    # combined outputs
    if all_summaries:
        combined = pd.concat(all_summaries, ignore_index=True)
        combined.to_csv(os.path.join(SUMMARY_DIR, 'all_arrays_contributions.csv'), index=False)
        dR2_cols = [c for c in combined.columns if c.startswith('dR2::')]
        wide = combined[['array', 'session', 'channel'] + dR2_cols].copy()
        wide.columns = ['array', 'session', 'channel'] + [c.split('::')[1] for c in dR2_cols]
        wide.to_csv(os.path.join(SUMMARY_DIR, 'all_arrays_dR2_by_family.csv'), index=False)
        print(f"\nWrote combined summaries -> {SUMMARY_DIR}")
    else:
        print("\nNo summaries produced.")
