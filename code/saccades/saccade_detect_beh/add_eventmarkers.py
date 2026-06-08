"""
Attach iRec eventmarkers to the saccade dataset, as a full-length marker channel.

The saccade dataset (stitched_sessions.npz) is the raw iRec eye signal at 500 Hz:
one sample per row of the original iRec position file ({eye}.csv), NaN-padded back
to the original length. No conversion was applied, so stitched sample i corresponds
to row i of that position file.

The eventmarkers live in {eye}net.csv (col 0 = iRec CPU timestamp, col 1 = event id),
on the SAME iRec clock as the position file's time column (col 0). We locate each
marker's timestamp inside the position file's time column with np.searchsorted (the
same operation as tc.match_irec_times) to get its sample index into the stitched arrays.

Output: saccade_eventmarkers.npz, per session:
    {session}__evt_channel - object array, length == len(x_nan/pred_nan/...). Element i is
                             an int array of all marker ids that landed on sample i (empty
                             where no marker). Aligned 1:1 with the eye/saccade traces.
    {session}__evt_idx     - sparse: sample index of each marker (convenience)
    {session}__evt_id      - sparse: marker id at each evt_idx (convenience)
plus 'sessions' and 'fs'. Load with allow_pickle=True (object array).
"""

import os
import glob
import collections
import numpy as np
import pandas as pd

EYE_DIR  = "/cs/projects/MWzeronoise/Analysis/4Shivangi/Datasets/eye_data"
SACC_NPZ = "/cs/projects/MWzeronoise/Analysis/4Shivangi/Results/saccade_detection/stitched_sessions.npz"
OUT_NPZ  = "/cs/projects/MWzeronoise/Analysis/4Shivangi/Results/saccade_detection/saccade_eventmarkers.npz"

STROBE_ID = 996  # dropped


def find_eye_files(session):
    """Return (pos_csv, net_csv) for a session, matching mapping_corners.py logic."""
    folder = os.path.join(EYE_DIR, session)
    csv_files = glob.glob(os.path.join(folder, "*.csv"))
    eye_file = None
    for csv_file in csv_files:
        base = os.path.basename(csv_file)
        if "net.csv" not in base:
            eye_file = base.replace(".csv", "")
    if eye_file is None:
        raise FileNotFoundError(f"No position .csv found in {folder}")
    pos_csv = os.path.join(folder, eye_file + ".csv")
    net_csv = os.path.join(folder, eye_file + "net.csv")
    return pos_csv, net_csv


def main():
    sacc = np.load(SACC_NPZ, allow_pickle=True)
    sessions = [str(s).strip() for s in sacc["sessions"]]

    out = {}
    missing = []
    for session in sessions:
        if not os.path.isdir(os.path.join(EYE_DIR, session)):
            print(f"{session}: no eye_data folder -- skipped (no eventmarkers attached)")
            missing.append(session)
            continue
        pos_csv, net_csv = find_eye_files(session)

        # iRec position file: col 0 = iRec CPU time, one row per 500 Hz sample
        pos_t = pd.read_csv(pos_csv, usecols=[0]).to_numpy().ravel()

        # eventmarkers: col 0 = iRec CPU time (same clock), col 1 = event id
        evt = pd.read_csv(net_csv, usecols=[0, 1]).to_numpy()
        evt_t = evt[:, 0]
        evt_id = evt[:, 1].astype(np.int64)

        # locate each marker timestamp in the position-file time column -> sample index
        evt_idx = np.searchsorted(pos_t, evt_t)

        # stitched length matches the position-file length (1:1, NaN-padded to orig_len)
        n_stitch = len(sacc[f"{session}__x_nan"])

        # keep markers inside the recording and drop the 996 strobe
        keep = (evt_idx >= 0) & (evt_idx < n_stitch) & (evt_id != STROBE_ID)
        evt_idx = evt_idx[keep]
        evt_id = evt_id[keep]

        # dense channel aligned 1:1 with the eye/saccade traces:
        # element i = array of all marker ids on sample i (shared empty array where none)
        empty = np.empty(0, dtype=np.int64)
        chan = np.empty(n_stitch, dtype=object)
        chan.fill(empty)
        by_idx = collections.defaultdict(list)
        for ix, mid in zip(evt_idx, evt_id):
            by_idx[ix].append(mid)
        for ix, ids in by_idx.items():
            chan[ix] = np.array(ids, dtype=np.int64)

        out[f"{session}__evt_channel"] = chan
        out[f"{session}__evt_idx"] = evt_idx.astype(np.int64)
        out[f"{session}__evt_id"] = evt_id

        max_per_sample = max((len(v) for v in by_idx.values()), default=0)
        print(f"{session}: {len(evt_idx)} markers on {len(by_idx)} samples "
              f"(<=996 dropped, max {max_per_sample}/sample; "
              f"len={n_stitch}, pos_len={len(pos_t)})")

    out["sessions"] = [s for s in sessions if s not in missing]
    out["fs"] = int(sacc["fs"])
    np.savez(OUT_NPZ, **out)
    print(f"\nSaved {OUT_NPZ}")
    if missing:
        print(f"WARNING: no eventmarkers for {missing} (no eye_data folder found)")


if __name__ == "__main__":
    main()
