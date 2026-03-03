# code to check if the RF centers fall inside the stimulus outlines

#### Code instructions #######################################################################################################
# 1. run mapping_corner.py for stimuli corner in cartesian coordinates
# 2. run prepare_RF.py for Rfs in cartseian coordinates
# 3. order the corner using corner_order.py and run warping_stim.py to warp the stimuli in the corners we got in step 1
# 4. Check if RF falls inside the warped stim using RFarea_stim.py
# 5. Check how everything looks using check_RF_in_stim.ipynb
##############################################################################################################################

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(1, '/mnt/cs/projects/MWzeronoise/Analysis/4Shivangi/code/functions/unreal_logfile')
from parse_logfile import TextLog
import pandas as pd
from skimage.transform import ProjectiveTransform
from skimage import io
from skimage.filters import threshold_otsu
from skimage.measure import find_contours
from scipy import interpolate
import pickle
from shapely.geometry import Polygon, Point
from concurrent.futures import ThreadPoolExecutor, as_completed
import h5py
import csv
import random

# Load RF data
os.chdir('/cs/projects/MWzeronoise/Analysis/4Shivangi/Results/RF_VR_mapping/RFs/20230209')
rf_radius = np.load('RF_radius.npy')
center_coords = np.load('center_coords.npy')

# Check RF area overlap with polygon

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

# Input/output HDF5 paths
input_hdf5_file = '/cs/projects/MWzeronoise/Analysis/4Shivangi/Results/RF_VR_mapping/stim_outline/20230214_try/processed_data.h5'
output_hdf5_file = '/cs/projects/MWzeronoise/Analysis/4Shivangi/Results/RF_VR_mapping/RFarea_stim/20230214_try/RF_stim_area.h5'
os.makedirs('/cs/projects/MWzeronoise/Analysis/4Shivangi/Results/RF_VR_mapping/RFarea_stim/20230214_try', exist_ok=True)
results = []

with h5py.File(input_hdf5_file, 'r') as infile, h5py.File(output_hdf5_file, 'w') as outfile:
    trial_names = list(infile.keys())
    for trial_name in trial_names:
        print(f"Processing Trial: {trial_name}")

        try:
            trial_group = infile[trial_name]
            trial_result_group = outfile.create_group(trial_name)

            trial_result_group.attrs['A_Identity'] = trial_group.attrs.get('A_Identity', 'Unknown')
            trial_result_group.attrs['B_Identity'] = trial_group.attrs.get('B_Identity', 'Unknown')

            time_point_names = list(trial_group.keys())
            for time_point_name in time_point_names:
                time_point_group = trial_group[time_point_name]
                tp_result_group = trial_result_group.create_group(time_point_name)

                transformed_outline_A = time_point_group.get('transformed_outline_A', None)
                transformed_outline_B = time_point_group.get('transformed_outline_B', None)

                outline_A = transformed_outline_A[()] if transformed_outline_A is not None else None
                outline_B = transformed_outline_B[()] if transformed_outline_B is not None else None

                polyA = outline_A.tolist() if outline_A is not None and len(outline_A) >= 3 else []
                polyB = outline_B.tolist() if outline_B is not None and len(outline_B) >= 3 else []

                for idx, row in enumerate(center_coords):
                    center = row[:2]
                    rad = rf_radius[idx]
                    if np.isnan(center).any() or np.isnan(rad):
                        continue

                    inside_A = is_overlap_at_least(center, rad, polyA)
                    inside_B = is_overlap_at_least(center, rad, polyB)

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
                        'InsideOutlineB': inside_B
                    })

        except Exception as e:
            print(f"Failed trial {trial_name}: {e}")

print("Summary of results:")
for result in results:
    print(result)