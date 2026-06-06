"""
background_parsing.py
=====================
Goal: for every BACKGROUND time point (RF not on the leaf/apple), decide
whether each RF is pointing at SKY, MOUNTAIN, or GRASS.

The background (sky / mountain / grass) is a static skybox. The mountain is far away and
never changes size, so it occupies a FIXED elevation band on the dome. We pin that band
once, from the right leaf at its first time point, because there the mountain sits inside
the leaf's vertical span. Measured tip-to-tip along the leaf:

        upper leaf tip --> mountain top   = 12.7 units
        mountain top   --> mountain base  =  6.5 units
        mountain base  --> lower leaf tip =  9.0 units
        ------------------------------------------------
        full leaf height (tip to tip)     = 28.2 units

so the mountain is a known PROPORTION of the leaf. Knowing the leaf's tip elevations in
degrees turns those proportions into real mountain-top / mountain-base elevations.

Pipeline (sections below):
  1. leaf tip fractions inside the warped bounding box (from the texture).
  2. parse the session log: stim locations/sizes, leaf identities, and eye positions.
  3. calibrate the mountain elevation band from the right leaf at its first time point.
  4. map an RF (retinal frame) to its dome elevation  -- the inverse of dome2eye.
  5. walk the overlap .h5, classify every background RF, and summarise per channel.

Frames: center_coords (RF) and stimulus outlines are in the gaze-relative RETINAL frame;
dome2eye maps dome(world) -> retinal, so we invert it to get the RF's world elevation.
Because we classify by ELEVATION only, trackball heading (a yaw of the world) does not
matter -- yaw leaves elevations unchanged. Run in the warping env.
"""

import os
import sys
import glob
import pickle
from collections import defaultdict
from datetime import datetime

import numpy as np
import h5py

sys.path.insert(1, '/mnt/cs/projects/MWzeronoise/Analysis/4Shivangi/code/functions/eyetracking')
sys.path.insert(1, '/mnt/cs/projects/MWzeronoise/Analysis/4Shivangi/code/functions/convert_unreal_coordinates')
sys.path.insert(1, '/mnt/cs/projects/MWzeronoise/Analysis/4Shivangi/code/functions/unreal_logfile')
import time_conversion as tc
from irec_conversion import dome2cartesian, cartesian2dome, normalize
from dome_conversion import calc_irec_rotation, eulerRodriguesVectorRotation
from convert_unreal_coordinates import relative_spherical
from parse_logfile import TextLog
from skimage import io
from skimage.filters import threshold_otsu

# --------------------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------------------- #
session = '20230214'
RF_sessions = ['20230202', '20230206', '20230209']   # sessions with gaussian RF fits

# proportions along the leaf, tip to tip (same units as measured on the screen)
P_TOP = 12.7    # upper leaf tip -> mountain top
P_MTN = 6.5     # mountain top   -> mountain base
P_BOT = 9.0     # mountain base  -> lower leaf tip
P_TOTAL = P_TOP + P_MTN + P_BOT

n_stimuli = 5             # ImageStimulus objects logged per trial
RIGHT_STIM_OFFSET = 2     # right-side leaf == stim A == index (trial*n_stimuli + 2)
eye_coords = np.array([1.5, 2.93, -13.77])   # head position, same as mapping_corners
DOME_R = 60.0

base = '/cs/projects/MWzeronoise/Analysis/4Shivangi'
stim_dir = f'{base}/Datasets/eye_data/{session}/stimuli'
image_paths = {49: f'{stim_dir}/hgblsp_049.png', 51: f'{stim_dir}/hgblsp_051.png'}
overlap_h5 = f'{base}/Results/RF_VR_mapping/RFarea_stim/{session}/RF_stim_collapse.h5'
out_h5 = f'{base}/Results/RF_VR_mapping/RFarea_stim/{session}/RF_background.h5'   # mirrors overlap_h5
out_path = f'{base}/Results/RF_VR_mapping/RFarea_stim/{session}/RF_background.pkl'  # calibration + summary


# --------------------------------------------------------------------------------------- #
# SECTION 1 : where the leaf tips sit inside the bounding box (the warped texture quad)
# --------------------------------------------------------------------------------------- #
# The texture fills the quad, but the leaf has blank margin above its top tip and below its
# stem, so the box edges are NOT the tips. Return tip positions as fractions of the box
# height, measured from the top edge (image row 0 == top / higher-elevation edge).
def leaf_tip_fractions(image_path):
    im = io.imread(image_path)
    if im.ndim == 3 and im.shape[2] == 4:
        mask = im[:, :, 3] > 0                      # leaf = non-transparent
    else:
        g = io.imread(image_path, as_gray=True)
        mask = g < threshold_otsu(g)               # leaf = darker than the white background
    H = mask.shape[0]
    rows = np.where(mask.any(axis=1))[0]
    return rows.min() / H, rows.max() / H          # f_top, f_bot


# --------------------------------------------------------------------------------------- #
# SECTION 2 : parse the session log -> stim locations/sizes, leaf identities, eye positions
# --------------------------------------------------------------------------------------- #
def load_session(filename):
    stim_loc, stim_ts, stim_params = [], [], []
    with TextLog(filename) as log:
        indx = [ii for ii, name in enumerate(log.all_ids['name']) if name.startswith('ImageStimulus')]
        for ii, istim in enumerate(indx):
            if ii + n_stimuli == len(indx):
                break
            this_id = log.all_ids[istim]
            next_id = log.all_ids[indx[ii + n_stimuli]]
            loc, pos_ts = log.parse_spherical(obj_id=this_id['id'], st=this_id['start'], end=next_id['start'])
            params = log.parse_initial_parameters(obj_id=this_id['id'], st=this_id['start'], end=next_id['start'])
            stim_loc.append(loc)
            stim_ts.append(pos_ts)
            stim_params.append(params)
        trial_info = log.get_info_per_trial(return_eventmarkers=True)

    stim_width = np.array([np.uint16(p['Scale'] * 200) for p in stim_params])
    vertical_offset = np.array([np.uint16(p['Height']) for p in stim_params])
    # the right-side leaf is the target when the target is on the right, else the distractor
    right_identity = np.where(trial_info['Right'] == 1,
                              trial_info['MorphTarget'], trial_info['MorphDistractor'])
    return stim_loc, stim_ts, stim_width, vertical_offset, right_identity


# eye position per trial, sampled at exactly the target-leaf (+3) time points -> one row per
# h5 time point. Mirrors the eye section of mapping_corners.py.
def eye_per_trial(filename, stim_ts):
    folder = os.path.dirname(filename) + '/'
    eye_file = next(os.path.basename(f).replace('.csv', '')
                    for f in glob.glob(folder + '*.csv') if 'net.csv' not in os.path.basename(f))
    log_irec_offset = tc.align_irec(filename, folder + eye_file + 'net.csv')
    irec_pos = np.genfromtxt(folder + eye_file + '.csv', delimiter=',', skip_header=1)
    eye_evt = np.genfromtxt(folder + eye_file + 'net.csv', delimiter=',', skip_header=1)

    trl_starts = eye_evt[:, 0][eye_evt[:, 1] == 3000]
    eye_trl = []
    for ist, _ in enumerate(trl_starts):
        if ist * n_stimuli >= len(stim_ts):
            break
        eye_x, eye_y, _ = tc.irec2log(irec_pos[:, 1], irec_pos[:, 2], irec_pos[:, 0],
                                      stim_ts[ist * n_stimuli + 3], log_irec_offset)
        eye_trl.append(np.stack((eye_x, eye_y), 1))
    return eye_trl


# --------------------------------------------------------------------------------------- #
# SECTION 3 : calibrate the mountain elevation band from the right leaf (first time point)
# --------------------------------------------------------------------------------------- #
# Elevation of a world direction, as seen from the eye -- the same projection dome2eye uses
# (dome point at R=60, minus eye, then read elevation). Used for both the leaf and the RF so
# the two are directly comparable.
def direction_to_elevation(vecs):
    return cartesian2dome(normalize(vecs) * DOME_R)[1]


def calibrate_mountain(stim_loc, stim_width, vertical_offset, itrl, f_top, f_bot):
    idx = itrl * n_stimuli + RIGHT_STIM_OFFSET
    x, y, z = stim_loc[idx].T                                   # azimuth, elevation, radius vs time
    a, e = relative_spherical.find_stimulus_corners(
        azimuth=x, elevation=y, radius=z,
        width=stim_width[idx], height=stim_width[idx], vertical_offset=vertical_offset[idx])
    # four box corners at the first time point -> elevation as seen from the eye.
    # dome2cartesian gives the corner's POSITION relative to the dome centre; subtracting
    # eye_coords turns it into a DIRECTION FROM THE EYE (the leaf is at finite distance, so
    # this parallax matters). This matches the RF in Section 4, which is also a direction from
    # the eye in dome axes -- same viewpoint, same axes, so the elevations are comparable.
    corner_dirs = dome2cartesian(a[:, 0], e[:, 0]) - eye_coords
    corner_elev = direction_to_elevation(corner_dirs)
    box_top, box_bot = corner_elev.max(), corner_elev.min()

    H_box = box_top - box_bot
    leaf_top = box_top - f_top * H_box                          # actual leaf tips (not box edges)
    leaf_bot = box_top - f_bot * H_box
    d_leaf = leaf_top - leaf_bot
    mtn_top = leaf_top - (P_TOP / P_TOTAL) * d_leaf
    mtn_bot = leaf_top - ((P_TOP + P_MTN) / P_TOTAL) * d_leaf
    return dict(box_top=box_top, box_bot=box_bot, leaf_top=leaf_top, leaf_bot=leaf_bot,
                mtn_top=mtn_top, mtn_bot=mtn_bot)


# --------------------------------------------------------------------------------------- #
# SECTION 4 : map RFs (retinal frame) to their dome elevation  -- inverse of dome2eye
# --------------------------------------------------------------------------------------- #
# Two independent things define a "frame": the AXES (gaze-aligned vs dome-aligned) and the
# VIEWPOINT/origin (measured from the eye vs from the dome centre). Here we change AXES only.
#
# center_coords are unit retinal-cartesian vectors (gaze-aligned axes). dome2eye rotated
# world directions into that frame using the eye direction; eulerRodriguesVectorRotation
# applies the OPPOSITE rotation, re-expressing each RF in DOME AXES. We do NOT move the RF to
# the dome centre: it stays a direction emanating from the EYE (a line of sight has no end
# point, so there is no parallax to apply). So the RF ends up as "a direction from the eye,
# in dome axes" -- the same space the leaf is put into in Section 3 (see the - eye_coords
# step there, which turns the leaf's finite dome-centred point into a direction from the eye).
# Both being eye-viewpoint + dome-axes is what makes the elevation comparison valid.
def rf_dome_elevation(rf_xyz, irec_x, irec_y):
    rot_axes, theta = calc_irec_rotation(np.array([irec_x]), np.array([irec_y]))
    dome_dir = eulerRodriguesVectorRotation(rot_axes, theta, rf_xyz)   # (N,3)
    return direction_to_elevation(dome_dir)


def classify(elev, mtn_top, mtn_bot):
    label = np.full(elev.shape, 'grass', dtype=object)
    label[elev > mtn_top] = 'sky'
    label[(elev <= mtn_top) & (elev >= mtn_bot)] = 'mountain'
    return label


# --------------------------------------------------------------------------------------- #
# SECTION 5 : walk the overlap .h5, classify background RFs, summarise per channel
# --------------------------------------------------------------------------------------- #
def main(max_trials=None):
    folder = f'//cs/projects/MWzeronoise/Analysis/4Shivangi/Datasets/eye_data/{session}/'
    formatted_date = f'{session[:4]}_{session[4:6]}_{session[6:]}'
    filename = glob.glob(folder + f'{formatted_date}*.log')[0]

    stim_loc, stim_ts, stim_width, vertical_offset, right_identity = load_session(filename)
    eye_trl = eye_per_trial(filename, stim_ts)

    # pick the closest RF session and load the 3D RF centers (the .h5 only stores x,y)
    fmt = lambda s: datetime.strptime(s, '%Y%m%d')
    closest_RF = min(RF_sessions, key=lambda rf: abs((fmt(rf) - fmt(session)).days))
    center_coords = np.load(f'{base}/Results/RF_VR_mapping/RFs/center_radius/{closest_RF}/center_coords.npy')

    # --- calibrate the mountain band on an example trial with leaf 49/51 on the right ---
    n_trials = min(len(stim_loc) // n_stimuli, len(right_identity))
    example = next(t for t in range(n_trials) if right_identity[t] in (49, 51))
    leaf_id = int(right_identity[example])
    f_top, f_bot = leaf_tip_fractions(image_paths[leaf_id])
    cal = calibrate_mountain(stim_loc, stim_width, vertical_offset, example, f_top, f_bot)
    mtn_top, mtn_bot = cal['mtn_top'], cal['mtn_bot']

    print(f'calibration trial {example}, right leaf {leaf_id}, tip fracs ({f_top:.3f},{f_bot:.3f})')
    print(f'  leaf tips elev [{cal["leaf_bot"]:.2f}, {cal["leaf_top"]:.2f}]  ->  '
          f'MOUNTAIN band elev [{mtn_bot:.2f}, {mtn_top:.2f}]')
    print(f'  sky: elev>{mtn_top:.2f} | mountain: {mtn_bot:.2f}..{mtn_top:.2f} | grass: elev<{mtn_bot:.2f}')

    # --- classify every background RF; write an h5 that mirrors overlap_h5 exactly ---
    # The output uses the SAME trial / time_point / Point group names as RF_stim_collapse.h5,
    # so trial 1 / time_point 3 / channel 4 that is background there has its sky/mountain/grass
    # label at the identical path here. Only background points are written (points inside A/B/C
    # are skipped, just like the overlap test labels them as on-stimulus).
    counts = defaultdict(lambda: dict(sky=0, mountain=0, grass=0, unknown=0))   # per channel
    with h5py.File(overlap_h5, 'r') as f, h5py.File(out_h5, 'w') as of:
        of.attrs['mtn_top'] = mtn_top
        of.attrs['mtn_bot'] = mtn_bot
        of.attrs['calibration_trial'] = int(example)
        of.attrs['calibration_leaf'] = int(leaf_id)
        trials = sorted(f.keys(), key=lambda n: int(n.split('_')[-1]))
        if max_trials:
            trials = trials[:max_trials]
        for tname in trials:
            itrl = int(tname.split('_')[-1])
            tg = f[tname]
            for tpname in tg.keys():
                j = int(tpname.split('_')[-1])
                tpg = tg[tpname]

                # background points = RF outside every stimulus (A, B, C all False)
                pnames, idxs = [], []
                for pname in tpg.keys():
                    pg = tpg[pname]
                    if pg['inside_transformed_outline_A'][()] or \
                       pg['inside_transformed_outline_B'][()] or \
                       pg['inside_transformed_outline_C'][()]:
                        continue
                    pnames.append(pname)
                    idxs.append(int(pname.split('_')[1]) - 1)
                if not idxs:
                    continue

                # placing the RF on the dome needs this time point's eye position; if it is
                # missing/NaN we still record the point but label it 'unknown' so nothing is lost
                eye_ok = itrl < len(eye_trl) and j < len(eye_trl[itrl])
                if eye_ok:
                    irec_x, irec_y = eye_trl[itrl][j]
                    eye_ok = bool(np.isfinite([irec_x, irec_y]).all())
                if eye_ok:
                    elev = rf_dome_elevation(center_coords[idxs], irec_x, irec_y)
                    labels = classify(elev, mtn_top, mtn_bot)
                else:
                    elev = np.full(len(idxs), np.nan)
                    labels = np.array(['unknown'] * len(idxs), dtype=object)

                tp_out = of.require_group(tname).require_group(tpname)
                for pname, lab, el in zip(pnames, labels, elev):
                    pg_out = tp_out.create_group(pname)
                    pg_out.create_dataset('background', data=np.bytes_(str(lab)))
                    pg_out.create_dataset('elevation', data=float(el))
                    counts['_'.join(pname.split('_')[:2])][lab] += 1

    # --- summarise: proportion of background time each channel is in each band ---
    summary = {}
    for chan, c in counts.items():
        tot = c['sky'] + c['mountain'] + c['grass']
        summary[chan] = {k: (c[k] / tot if tot else 0.0) for k in ('sky', 'mountain', 'grass')}
    with open(out_path, 'wb') as fh:
        pickle.dump(dict(calibration=cal, mtn_top=mtn_top, mtn_bot=mtn_bot,
                         counts=dict(counts), summary=summary), fh)
    print(f'per-point labels -> {out_h5}')
    print(f'saved {len(summary)} channels -> {out_path}')
    for chan in list(summary)[:5]:
        s = summary[chan]
        print(f'  {chan}: sky {s["sky"]:.2f}  mountain {s["mountain"]:.2f}  grass {s["grass"]:.2f}')
    return summary


if __name__ == '__main__':
    main()
