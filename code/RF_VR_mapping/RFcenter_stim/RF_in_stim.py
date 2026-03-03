# code to check if the RF centers fall inside the stimulus outlines

#### Code instructions #######################################################################################################
# 1. run mapping_corner.py for stimuli corner in cartesian coordinates
# 2. run prepare_RF.py for Rfs in cartseian coordinates
# 3. order the corner using corner_order.py and run warping_stim.py to warp the stimuli in the corners we got in step 1
# 4. Check if RF falls inside the warped stim using RF_in_stim.py
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
from scipy import interpolate                       #run this in warping env
import pickle
from shapely.geometry import Polygon, Point
from concurrent.futures import ThreadPoolExecutor, as_completed
import h5py
import csv


# import all the RFs -------------------------

RF_file = '/cs/projects/MWzeronoise/Analysis/4Shivangi/Results/RF_VR_mapping/RFs/cleaned_RF_cartesian.csv'

# Function to extract and format (x, y) values 
def extract_xy_values(csv_file):
    xy_pairs = []
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)  
        for row in reader:
            # Extract x and y, round to 3 decimal places, and store as tuple
            x = round(float(row['x']), 3)
            y = round(float(row['y']), 3)
            xy_pairs.append((x, y))
    return xy_pairs

# Call the function and print the result
xy_values = extract_xy_values(RF_file)
test_points = xy_values #list of all the RF centers

# check if the RF falls inside the stim outline ----------------------

# Function to check if points lie within the polygon
def check_points_within_polygon(polygon_coords, test_points):
    if polygon_coords is None or len(polygon_coords) < 3:
        return [False] * len(test_points)

    if not np.array_equal(polygon_coords[0], polygon_coords[-1]):
        polygon_coords = np.vstack([polygon_coords, polygon_coords[0]])

    try:
        polygon = Polygon(polygon_coords)
        results = [polygon.contains(Point(point)) for point in test_points]
        return results
    except Exception as e:
        print(f"Error creating polygon: {e}")
        return [False] * len(test_points)

# Path to the HDF5 file containing transformed outlines
input_hdf5_file = '/cs/projects/MWzeronoise/Analysis/4Shivangi/Results/RF_VR_mapping/stim_outline/20230214/processed_data.h5'

# Path to save the results in HDF5 format
output_hdf5_file = '/cs/projects/MWzeronoise/Analysis/4Shivangi/Results/RF_VR_mapping/RF_in_stim/20230214/RF_stim_check.h5'

# Read input data and process
results = []

with h5py.File(input_hdf5_file, 'r') as infile, h5py.File(output_hdf5_file, 'w') as outfile:
    for trial_name in infile.keys():
        print(f"Processing Trial: {trial_name}")
        trial_group = infile[trial_name]
        trial_result_group = outfile.create_group(trial_name)  # Create a group in the output file

        # Access metadata for the trial
        trial_result_group.attrs['A_Identity'] = trial_group.attrs.get('A_Identity', 'Unknown')
        trial_result_group.attrs['B_Identity'] = trial_group.attrs.get('B_Identity', 'Unknown')

        for time_point_name in trial_group.keys():
            time_point_group = trial_group[time_point_name]
            tp_result_group = trial_result_group.create_group(time_point_name)

            # Access the outlines
            transformed_outline_A = time_point_group.get('transformed_outline_A', None)
            transformed_outline_B = time_point_group.get('transformed_outline_B', None)

            # Convert HDF5 datasets to numpy arrays
            outline_A = transformed_outline_A[()] if transformed_outline_A is not None else None
            outline_B = transformed_outline_B[()] if transformed_outline_B is not None else None

            # Check each test point against the polygons and store the results
            inside_A = check_points_within_polygon(outline_A, test_points) if outline_A is not None else [False] * len(test_points)
            inside_B = check_points_within_polygon(outline_B, test_points) if outline_B is not None else [False] * len(test_points)

            # Save results for each test point
            for idx, point in enumerate(test_points):
                point_result_group = tp_result_group.create_group(f'Point_{idx + 1}_{point}')

                point_result_group.create_dataset('test_point', data=point)
                point_result_group.create_dataset('inside_transformed_outline_A', data=inside_A[idx])
                point_result_group.create_dataset('inside_transformed_outline_B', data=inside_B[idx])

                # Append results for debugging or additional processing
                results.append({
                    'Trial': trial_name,
                    'TimePoint': time_point_name,
                    'TestPoint': point,
                    'InsideOutlineA': inside_A[idx],
                    'InsideOutlineB': inside_B[idx]
                })

# Print summary of results
print("Summary of results:")
for result in results:
    print(result)

