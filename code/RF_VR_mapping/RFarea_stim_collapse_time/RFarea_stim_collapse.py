# code to check if the RF centers fall inside the stimulus outlines accounting for collapse

"""
#### Code instructions #######################################################################################################
# 1. run mapping_corner.py for stimuli corner in cartesian coordinates
# 2. run prepare_RF.py for Rfs in cartseian coordinates
# 3. order the corner using corner_order.py and run warping_stim.py to warp the stimuli in the corners we got in step 1
# 4. Check if RF falls inside the warped stim accounting for collapse using RFarea_stim_collapse.py
# 5. Check how everything looks using RFoverlap_perc_collapse.py
##############################################################################################################################
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
import pickle
import pandas as pd
from scipy import interpolate
from skimage import io
from skimage.filters import threshold_otsu
from skimage.measure import find_contours
from shapely.geometry import Polygon, Point
from skimage.transform import ProjectiveTransform
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime #run this in warping env

sessions = ['20230213'] #,'20230203', '20230208', '20230209','20230214']
RF_sessions = ['20230202', '20230206', '20230209']

base_path = '/cs/projects/MWzeronoise/Analysis/4Shivangi/Results/RF_VR_mapping/RFs/center_radius'

def str_to_date(s):
    return datetime.strptime(s, "%Y%m%d")

RF_dates = {rf: str_to_date(rf) for rf in RF_sessions}

for session in sessions:
    session
    session_date = str_to_date(session)

    # find closest RF session
    closest_RF = min(
        RF_sessions,
        key=lambda rf: abs((RF_dates[rf] - session_date).days)
    )
    rf_path = os.path.join(base_path, closest_RF)
    rf_path
    os.chdir(rf_path)

    rf_radius = np.load('RF_radius.npy')
    center_coords = np.load('center_coords.npy')

    #-----------Overlap and Stim Collapse---------#

    ## Check RF area overlap with polygon 
    def is_overlap_at_least(circle_center, radius, polygon_coords, threshold=0.01):
        if not polygon_coords or len(polygon_coords) < 3:
            return False
        try:
            circle = Point(circle_center).buffer(radius)
            target_polygon = Polygon(polygon_coords)
            if target_polygon.area == 0:
                return False
            intersection = circle.intersection(target_polygon)
            circle_area = circle.area
            polygon_area = target_polygon.area 
            intersection_area = intersection.area
            if min(circle_area, polygon_area) == 0:
                return False
            overlap_percentage = (intersection_area / min(circle_area, polygon_area)) * 100
            return overlap_percentage >= threshold
        except Exception as e:
            print(f"Error checking overlap: {e}")
            return False

    def is_collapsed(coords, var_thresh=1.0, height_thresh=10.0):
        """
        Returns True if:
        - Height is above threshold AND variance in X or Y ≥ threshold (unstable/expanded)
        - OR outline is line-like: small area + large height
        """
        if coords is None or len(coords) == 0:
            return False  # nothing to check

        coords = np.array(coords)
        varX = np.var(coords[:, 0])
        varY = np.var(coords[:, 1])
        height = coords[:, 1].max() - coords[:, 1].min()

        # Clause: variance only matters if height is above threshold
        if height >= height_thresh and (varX >= var_thresh or varY >= var_thresh):
            return True

        # Line-like shape: large height alone
        if height >= height_thresh:
            return True

        return False

    ## Input/output HDF5 paths
    input_hdf5_file = f'/cs/projects/MWzeronoise/Analysis/4Shivangi/Results/RF_VR_mapping/stim_outline/{session}/processed_data.h5'
    output_hdf5_file = f'/cs/projects/MWzeronoise/Analysis/4Shivangi/Results/RF_VR_mapping/RFarea_stim/{session}/RF_stim_collapse.h5'
    output_dir = os.path.dirname(output_hdf5_file)
    os.makedirs(output_dir, exist_ok=True)
    results = []

    with h5py.File(input_hdf5_file, 'r') as infile, h5py.File(output_hdf5_file, 'w') as outfile:
        trial_names = sorted(infile.keys(), key=lambda name: int(name.split('_')[-1]))
        for trial_name in trial_names:
            print(f"Processing Trial: {trial_name}")
            try:
                trial_group = infile[trial_name]
                trial_result_group = outfile.create_group(trial_name)
                trial_result_group.attrs['A_Identity'] = trial_group.attrs.get('A_Identity', 'Unknown')
                trial_result_group.attrs['B_Identity'] = trial_group.attrs.get('B_Identity', 'Unknown')
                
                time_point_names = sorted(trial_group.keys(), key=lambda tp: int(tp.split('_')[-1]))
                for time_point_name in time_point_names:
                    time_point_group = trial_group[time_point_name]
                    tp_result_group = trial_result_group.create_group(time_point_name)

                    transformed_outline_A = time_point_group.get('transformed_outline_A', None)
                    transformed_outline_B = time_point_group.get('transformed_outline_B', None)

                    outline_A = transformed_outline_A[()] if transformed_outline_A is not None else None
                    outline_B = transformed_outline_B[()] if transformed_outline_B is not None else None

                    polyA = outline_A.tolist() if outline_A is not None and len(outline_A) >= 3 else []
                    polyB = outline_B.tolist() if outline_B is not None and len(outline_B) >= 3 else []

                
                    collapsed_A = is_collapsed(polyA, var_thresh=1.0, height_thresh=10.0, )
                    collapsed_B = is_collapsed(polyB, var_thresh=1.0, height_thresh=10.0, )

                    collapsed_case = collapsed_A or collapsed_B 

                # save only collapsed_case attribute
                    tp_result_group.attrs['collapsed_case'] = collapsed_case

                    # --- RF overlap checks ---
                    for idx, row in enumerate(center_coords):
                        center = row[:2]
                        rad = rf_radius[idx]
                        if np.isnan(center).any() or np.isnan(rad):
                            continue

                        # if collapse detected → force both inside_A and inside_B = True
                        if collapsed_case:
                            inside_A, inside_B = True, True
                        else:
                            inside_A = is_overlap_at_least(center, rad, polyA) if len(polyA) >= 3 else False
                            inside_B = is_overlap_at_least(center, rad, polyB) if len(polyB) >= 3 else False

                        # save to HDF5
                        center_str = f"{center[0]:.2f}_{center[1]:.2f}"
                        group_name = f"Point_{idx + 1}_{center_str}"
                        point_result_group = tp_result_group.create_group(group_name)
                        point_result_group.create_dataset('RF_center', data=center)
                        point_result_group.create_dataset('radius', data=rad)
                        point_result_group.create_dataset('inside_transformed_outline_A', data=inside_A)
                        point_result_group.create_dataset('inside_transformed_outline_B', data=inside_B)

                        results.append({
                            'Trial': trial_name,
                            'TimePoint': time_point_name,
                            'RF_center': center,
                            'Radius': rad,
                            'InsideOutlineA': inside_A,
                            'InsideOutlineB': inside_B,
                            'CollapsedCase': collapsed_case
                        })
            except Exception as e:
                print(f"Error processing {trial_name}: {e}")

    print("Summary of results:")
    for result in results:
        print(result)


