"""
=============================================================================================
Assemble a POOLED (multi-session) design + neural target per electrode  (step 3b, optional).
=============================================================================================
Pipeline position: OPTIONAL, between 'design' (step 3) and 'fit' (step 4). Only needed when you
want to fit ONE encoding model per electrode over SEVERAL sessions concatenated, instead of one
model per (session, channel). Running sessions SEPARATELY does NOT need this file at all -- the
normal FittingGLM.py / RegressorContributions.py path is unchanged.

For each channel it takes that electrode's per-session RAW design matrices + downsampled neural
targets (written by DesignMatrix.py) and stacks them into a single "virtual session" so the existing
fit / contributions steps can run on it with NO change to their file contract. Concretely it:

  1. loads each session's {sid}_dMatRaw_sparse.npz + _dMatRaw_metadata.npz + _neural_downsampled.npz
  2. ALIGNS columns across sessions by (family label, lag-within-family) -- raw kernels have a fixed
     width so this is exact; families/lags not present in EVERY session are dropped (logged loudly).
  3. Z-SCORES the neural target PER SESSION (each session zero-mean, unit-variance) so no single
     session's gain dominates the pooled ridge  [mitigation #4].
  4. optionally appends a SESSION-INDICATOR family (K-1 columns): a per-session intercept that soaks
     up any residual between-session offset  [mitigation #3; "add that we have different sessions"].
  5. stacks the aligned designs + neural, re-applies DesignMatrix.py's Part-2 processing (drop
     near-empty columns, L2-normalise) on the POOLED matrix, and writes, under a pooled "session"
     folder <pooled_session>/channel<ch>_regressors/results/:
        {pooled_session}_channel<ch>_dMatProcessed_sparse.npz / _metadata.npz   (+ session_id !)
        {pooled_session}_channel<ch>_neural_downsampled.npz
     The extra `session_id` row-vector in the metadata is what makes FittingGLM.py /
     RegressorContributions.py switch to session-aware CV folds + within-session permutation.

After running this, fit the pooled electrodes exactly like a normal session:
    FittingGLM.py / RegressorContributions.py with SESSIONS = [<pooled_session>]
(or let ArrayRun.py do it via POOL_SESSIONS=True).

Run in the warping env (needs acme).
"""

import os
import re
import sys
import glob
import warnings
from pathlib import Path

import numpy as np
from scipy.sparse import load_npz, save_npz, hstack, vstack, diags, csc_matrix

from acme import ParallelMap

warnings.filterwarnings("ignore", category=FutureWarning)

# make glm_config importable (single source of truth for the output tree)
for _d in (os.path.dirname(os.path.abspath(__file__)),
           os.path.dirname(os.path.dirname(os.path.abspath(__file__)))):
    if os.path.exists(os.path.join(_d, 'glm_config.py')):
        sys.path.insert(0, _d)
        break
from glm_config import RESULTS_DIR
results_dir = RESULTS_DIR

# -------------------------
# Config
# -------------------------
# Sessions to pool (in the order they should be concatenated). Set from ArrayRun.py when driven there.
SESSIONS_TO_POOL = ['20230203', '20230208', '20230209', '20230213', '20230214']
# Name of the virtual pooled "session" folder the artifacts are written under.
POOLED_SESSION = 'pooled_' + SESSIONS_TO_POOL[0] + '_' + SESSIONS_TO_POOL[-1]
CHANNELS = None   # None -> every channel present (raw design built) in ALL pooled sessions

# Append a per-session intercept (K-1 indicator columns) to absorb residual between-session offset.
ADD_SESSION_INDICATOR = True

# Drop near-empty pooled columns exactly like DesignMatrix.py Part-2 (col |sum| < this).
SPARSE_COL_MIN = 30

# SLURM / acme: one worker per channel. Assembly is I/O + a couple of sparse stacks -> light.
SLURM_PARTITION = '32GBS'
MAX_WORKERS = 100
MEM_PER_WORKER = '32GB'


# -------------------------
# Helpers
# -------------------------
def _raw_base(session, channel):
    return Path(results_dir) / session / f'channel{channel}_regressors' / 'results' / f'{session}_channel{channel}'


def channel_has_raw(session, channel):
    return os.path.exists(str(_raw_base(session, channel)) + '_dMatRaw_sparse.npz')


def discover_pooled_channels(sessions):
    """Channels whose RAW design exists in EVERY session to be pooled."""
    persess = []
    for s in sessions:
        chans = set()
        for p in glob.glob(str(Path(results_dir) / s / 'channel*_regressors')):
            m = re.search(r'channel(\d+)_regressors$', os.path.basename(p))
            if m and channel_has_raw(s, int(m.group(1))):
                chans.add(int(m.group(1)))
        persess.append(chans)
    return sorted(set.intersection(*persess)) if persess else []


def column_keys(regIdx, regLabels):
    """Per-column (label, lag-ordinal-within-family) key, in column order. regIdx is 1-based."""
    regIdx = np.asarray(regIdx).ravel()
    regLabels = [str(x) for x in np.asarray(regLabels).ravel()]
    keys, seen = [], {}
    for gid in regIdx:
        lab = regLabels[int(gid) - 1]
        seen[lab] = seen.get(lab, 0) + 1
        keys.append((lab, seen[lab] - 1))
    return keys


def load_session(session, channel):
    """Return (Rraw_csc, column-keys, neural-zscored, n_rows) for one session/channel, row-matched."""
    base = str(_raw_base(session, channel))
    R = load_npz(base + '_dMatRaw_sparse.npz').tocsc()
    m = np.load(base + '_dMatRaw_metadata.npz', allow_pickle=True)
    keys = column_keys(m['regIdx'], m['regLabels'])
    y = np.load(base + '_neural_downsampled.npz')['data'].astype(float).ravel()
    n = min(R.shape[0], y.shape[0])
    R, y = R[:n], y[:n]
    sd = y.std()
    y = (y - y.mean()) / (sd if sd > 0 else 1.0)      # per-session z-score  [mitigation #4]
    return R, keys, y, n


# -------------------------
# Per-channel worker: build the pooled artifacts for one electrode
# -------------------------
def assemble_channel(channel, sessions, pooled_session):
    """One acme worker: pool `sessions` for one electrode; write the pooled results/ artifacts."""
    sessions = list(sessions)
    have = [s for s in sessions if channel_has_raw(s, channel)]
    if len(have) < 2:
        print(f"ch{channel}: raw design in only {len(have)} session(s) -> nothing to pool, skip")
        return None

    loaded = {s: load_session(s, channel) for s in have}

    # --- column alignment: keep (label, lag) keys present in EVERY session, in session[0] order ---
    common = set.intersection(*[set(keys) for (_R, keys, _y, _n) in loaded.values()])
    order_keys = [k for k in loaded[have[0]][1] if k in common]
    if not order_keys:
        print(f"ch{channel}: no columns common to all sessions, skip")
        return None
    dropped = [k for k in loaded[have[0]][1] if k not in common]
    if dropped:
        fams = sorted({lab for lab, _ in dropped})
        print(f"ch{channel}: dropping {len(dropped)} unaligned columns across families {fams}")
    key_pos = {k: i for i, k in enumerate(order_keys)}

    # per-session selector -> reorder that session's columns into the common layout
    aligned = []
    session_id_parts = []
    for si, s in enumerate(have):
        R, keys, y, n = loaded[s]
        sel = np.full(len(order_keys), -1, dtype=int)
        for col, k in enumerate(keys):
            if k in key_pos:
                sel[key_pos[k]] = col
        aligned.append(R[:, sel])
        session_id_parts.append(np.full(n, si, dtype=int))
    R_pool = vstack(aligned).tocsc()                  # (sum rows) x len(order_keys)
    session_id = np.concatenate(session_id_parts)
    y_pool = np.concatenate([loaded[s][2] for s in have])
    K = len(have)

    # family label per common column, and the contiguous column blocks per family (layout order)
    labels = [lab for lab, _ in order_keys]
    blocks = []                                       # (label, [col indices])
    for c, lab in enumerate(labels):
        if blocks and blocks[-1][0] == lab:
            blocks[-1][1].append(c)
        else:
            blocks.append((lab, [c]))

    # --- assemble final column blocks: one family id per family, columns shared across sessions ---
    col_blocks, out_regIdx, out_labels = [], [], []
    gid = 0
    for lab, cols in blocks:
        gid += 1
        col_blocks.append(R_pool[:, cols])
        out_regIdx += [gid] * len(cols)
        out_labels.append(lab)

    # --- optional per-session intercept family (K-1 indicator columns)  [mitigation #3] ---
    if ADD_SESSION_INDICATOR and K > 1:
        gid += 1
        ind = np.zeros((len(session_id), K - 1))
        for si in range(1, K):                        # session 0 is the reference level
            ind[session_id == si, si - 1] = 1.0
        col_blocks.append(csc_matrix(ind))
        out_regIdx += [gid] * (K - 1)
        out_labels.append('session')

    R_pool = hstack(col_blocks).tocsc()
    regIdx = np.asarray(out_regIdx, dtype=int)        # 1-based family id per column
    regLabels = out_labels

    # --- Part-2 processing on the POOLED matrix (mirror DesignMatrix.py: drop near-empty + L2-norm) ---
    col_sum = np.abs(R_pool).sum(axis=0).A1
    keep = col_sum >= SPARSE_COL_MIN
    Rkeep = R_pool[:, keep]
    col_norm = np.sqrt(Rkeep.power(2).sum(axis=0)).A1
    col_norm[col_norm == 0] = 1.0
    R_proc = Rkeep @ diags(1 / col_norm)
    regIdx_kept = regIdx[keep]
    regLabelsNew = [regLabels[int(r) - 1] for r in np.unique(regIdx_kept)]
    if int((~keep).sum()):
        print(f"ch{channel}: dropped {int((~keep).sum())}/{keep.size} near-empty pooled columns")

    # frame_rate + per-family pre-window carried from session[0]'s PROCESSED metadata (by label)
    pbase = str(_raw_base(have[0], channel))          # .../{session}_channel{ch}
    pmeta = np.load(pbase + '_dMatProcessed_metadata.npz', allow_pickle=True)
    frame_rate = int(pmeta['frame_rate'])
    pre_by_lab = {str(l): int(p) for l, p in zip(np.asarray(pmeta['regLabels']).ravel(),
                                                 np.asarray(pmeta['lag_pre_frames']).ravel())}
    lag_pre_frames = np.array([pre_by_lab.get(l, 0) for l in regLabelsNew], dtype=int)

    # --- write pooled artifacts under the virtual pooled-session folder ---
    out_dir = Path(results_dir) / pooled_session / f'channel{channel}_regressors' / 'results'
    out_dir.mkdir(parents=True, exist_ok=True)
    pid = f'{pooled_session}_channel{channel}'
    save_npz(str(out_dir / f'{pid}_dMatProcessed_sparse.npz'), R_proc.tocsr())
    np.savez(str(out_dir / f'{pid}_dMatProcessed_metadata.npz'),
             regIdx=regIdx_kept, regLabels=np.array(regLabelsNew, dtype=object),
             lag_pre_frames=lag_pre_frames, frame_rate=frame_rate,
             session_id=session_id, session_names=np.array(have, dtype=object))
    np.savez(str(out_dir / f'{pid}_neural_downsampled.npz'),
             data=y_pool, frame_rate=frame_rate, session_id=session_id)
    print(f"ch{channel}: pooled {K} sessions {have} -> {R_proc.shape} rows, "
          f"{len(regLabelsNew)} families -> {out_dir}/{pid}_dMatProcessed_sparse.npz")
    return str(out_dir / f'{pid}_dMatProcessed_sparse.npz')


# -------------------------
# Driver: parallelise channels across the cluster (one worker per channel)
# -------------------------
if __name__ == '__main__':
    sessions = SESSIONS_TO_POOL
    channels = CHANNELS if CHANNELS is not None else discover_pooled_channels(sessions)
    if not channels:
        raise SystemExit(f'No channels have a raw design in all of {sessions} under {results_dir}')

    n_workers = min(MAX_WORKERS, len(channels))
    print(f"pooling {sessions} -> '{POOLED_SESSION}': {len(channels)} channels "
          f"({n_workers} workers on '{SLURM_PARTITION}')")

    with ParallelMap(assemble_channel, channels, sessions, POOLED_SESSION,
                     n_inputs=len(channels),
                     partition=SLURM_PARTITION,
                     n_workers=n_workers,
                     mem_per_worker=MEM_PER_WORKER,
                     setup_timeout=600,
                     write_worker_results=False,
                     setup_interactive=False) as pmap:
        pmap.compute()
