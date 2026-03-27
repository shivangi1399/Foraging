# warping the stimulus and saving the stimulus outline for each time point in a session

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
import sys
import gc
import h5py
import glob
import pandas as pd
from skimage.transform import ProjectiveTransform
from skimage import io
from skimage.filters import threshold_otsu
from skimage.measure import find_contours
from scipy import interpolate                       
import pickle
from shapely.geometry import Polygon
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
sys.path.insert(1, '/mnt/cs/projects/MWzeronoise/Analysis/4Shivangi/code/functions/unreal_logfile')
from parse_logfile import TextLog #run this in warping env

sessions = ['20230208'] #['20230209', '20230203', '20230208', '20230214']

for session in sessions:
  
    folder = f'//cs/projects/MWzeronoise/Analysis/4Shivangi/Datasets/eye_data/{session}/'
    formatted_date = f'{session[:4]}_{session[4:6]}_{session[6:]}'
    log_files = glob.glob(folder + f'{formatted_date}*.log')
    filename = log_files[0] 

    # stimulus identity info -----------------------------------------------------------------

    with TextLog(filename) as log:
        trial_data = log.get_info_per_trial(return_eventmarkers=True, return_loc=False)

    # Convert trial data to DataFrame
    df = pd.DataFrame(trial_data)

    # Add TargetIdentity column based on 'Right'
    df['TargetIdentity'] = np.where(df['Right'] == 1, 'A', 'B') #df['Right'] means the target was on the right. Stim A is always on right etc
    df['TargetIdentity_C'] = 'C'  # C is always present
    
    # Prepare a dictionary to store aggregated data
    aggregated_data = {
    'TrialIndex': [],
    'A_Identity': [],
    'B_Identity': [],
    'C_Identity': []
    }

    # Process each trial
    for trial_index in df['TrialIndex'].unique():
        trial_subset = df[df['TrialIndex'] == trial_index]
        
        # Collect morph targets for each identity
        a_morph_targets = trial_subset[trial_subset['TargetIdentity'] == 'A']['MorphTarget'].tolist()
        b_morph_targets = trial_subset[trial_subset['TargetIdentity'] == 'B']['MorphTarget'].tolist()
        c_morph_targets = trial_subset[trial_subset['TargetIdentity_C'] == 'C']['MorphTarget'].tolist()
        
        # Convert lists to space-separated strings
        a_identity_str = ' '.join(map(str, a_morph_targets))
        b_identity_str = ' '.join(map(str, b_morph_targets))
        c_identity_str = ' '.join(map(str, c_morph_targets))
        
        # Add to aggregated data
        aggregated_data['TrialIndex'].append(trial_index)
        aggregated_data['A_Identity'].append(a_identity_str)
        aggregated_data['B_Identity'].append(b_identity_str)
        aggregated_data['C_Identity'].append(c_identity_str)

    # Create the final DataFrame
    aggregated_df = pd.DataFrame(aggregated_data)

    # fill in the gaps in the df
    def fill_gaps(row):
        a_identity_str = row['A_Identity']
        b_identity_str = row['B_Identity']
        c_identity_str = row['C_Identity']

        # Convert strings to lists
        a_identity_list = a_identity_str.split() if a_identity_str else []
        b_identity_list = b_identity_str.split() if b_identity_str else []
        c_identity_list = c_identity_str.split() if c_identity_str else []
        
        # Define pairs
        pairs = {
            '30': '70',
            '70': '30',
            '49': '51',
            '51': '49'
        }
        
        # Fill B_Identity if A_Identity has a specific value
        if a_identity_list and not b_identity_list:
            # If A has specific values, fill B with corresponding pair
            b_identity_list = [pairs.get(a, '') for a in a_identity_list if pairs.get(a, '')]
        
        # Fill A_Identity if B_Identity has a specific value
        if b_identity_list and not a_identity_list:
            # If B has specific values, fill A with corresponding pair
            a_identity_list = [pairs.get(b, '') for b in b_identity_list if pairs.get(b, '')]
        
        # Convert lists back to space-separated strings
        a_identity_str_filled = ' '.join(a_identity_list)
        b_identity_str_filled = ' '.join(b_identity_list)
        c_identity_str_filled = ' '.join(c_identity_list)

        return pd.Series([a_identity_str_filled, b_identity_str_filled, c_identity_str_filled])

    # fill gaps
    aggregated_df[['A_Identity', 'B_Identity', 'C_Identity']] = aggregated_df.apply(fill_gaps, axis=1)
    stim_df = aggregated_df

    # Print the updated DataFrame
    print(stim_df)

    # -----------------------------------------------------------------------------------
    # transforming the stimuli outline to fit into the polygon coordinates of the stimuli  

    image_paths = {
        '30': '/cs/projects/MWzeronoise/Analysis/4Shivangi/Datasets/eye_data/20230214/stimuli/hgblsp_030.png',
        '49': '/cs/projects/MWzeronoise/Analysis/4Shivangi/Datasets/eye_data/20230214/stimuli/hgblsp_049.png',
        '51': '/cs/projects/MWzeronoise/Analysis/4Shivangi/Datasets/eye_data/20230214/stimuli/hgblsp_051.png',
        '70': '/cs/projects/MWzeronoise/Analysis/4Shivangi/Datasets/eye_data/20230214/stimuli/hgblsp_070.png',
        '100': '/cs/projects/MWzeronoise/Analysis/4Shivangi/Datasets/eye_data/20230214/stimuli/redapple_100.png',
    }

    # image processing functions ----
    def process_image(path, polygon_coords):
        def threshold_stimulus(image_path):
            image = io.imread(image_path)
            if image.shape[2] == 4:
                alpha_channel = image[:, :, 3]
                image_mask = alpha_channel > 0
                image_mask = np.pad(image_mask, pad_width=1, mode='constant', constant_values=0)
            else:
                image = io.imread(image_path, as_gray=True)
                image_mask = image > threshold_otsu(image)
                image_mask = np.pad(image_mask, pad_width=1, mode='constant', constant_values=1)
            return image, image_mask

        def extract_contour(image_mask):
            contours = find_contours(image_mask, level=0, fully_connected='low')
            largest_contour = max(contours, key=len)
            return largest_contour

        def normalize_contour(contour, image_shape):
            x = contour[:, 1] - 1
            y = contour[:, 0] - 1
            y = image_shape[0] - y
            x_norm = x / image_shape[1]
            y_norm = y / image_shape[0]
            normalized_contour = np.vstack((x_norm, y_norm)).T
            return normalized_contour

        def fitspline(contour, steps):
            x = contour[:, 0]
            y = contour[:, 1]
            tck, u = interpolate.splprep([x, y], k=3, s=0)
            u = np.linspace(0., 1., steps)
            outline = np.array(interpolate.splev(u, tck)).T
            return outline

        def transform_contour_to_polygon(contour, polygon_coords):
            if len(polygon_coords) < 4:
                print(f"Polygon coordinates must have at least 4 points. Provided: {polygon_coords}")
                return None

            if  not np.array_equal(polygon_coords[0], polygon_coords[-1]):
                polygon_coords = np.append(polygon_coords, [polygon_coords[0]], axis=0)

            polygon = Polygon(polygon_coords)
            src_points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
            dst_points = np.array(polygon.exterior.coords[:-1])
            transform = ProjectiveTransform()
            transform.estimate(src_points, dst_points)
            transformed_contour = transform(np.array(contour))
            return transformed_contour

        image, image_mask = threshold_stimulus(path)
        contour = extract_contour(image_mask)
        normalized_contour = normalize_contour(contour, image.shape[:2])
        outline = fitspline(normalized_contour, 1000)
        transformed_outline = transform_contour_to_polygon(outline, polygon_coords)
        return transformed_outline

    # reading in the reordered mesh corners ----
    os.chdir(f'/cs/projects/MWzeronoise/Analysis/4Shivangi/Results/RF_VR_mapping/reordered_corners/{session}/')
    with open('trials_reordered_list.pkl', 'rb') as file:
        trials_reordered_list = pickle.load(file)

    # warping according to the stimulus identity -----
    output_dir = f'/cs/projects/MWzeronoise/Analysis/4Shivangi/Results/RF_VR_mapping/stim_outline/{session}/'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'processed_data.h5')

    # Prepare lookup dictionary to avoid repetitive data access
    trial_lookup = {row['TrialIndex']: row for _, row in aggregated_df.iterrows()}

    # Function to process individual images for parallelization
    def process_image_parallel(path, coords):
        try:
            return process_image(path, coords)
        except Exception as e:
            print(f"Error processing image at {path}: {e}")
            return None

    # Define process_trial function to save memory
    def process_trial(trial_index, trial):
        print(f"Processing trial {trial_index}")
        row = trial_lookup.get(trial_index)
        if row is None:
            print(f"No matching trial in aggregated_df for trial_index {trial_index}")
            return None

        A_Identity = row['A_Identity'].split()[0] if row['A_Identity'] else None
        B_Identity = row['B_Identity'].split()[0] if row['B_Identity'] else None
        C_Identity = row['C_Identity'].split()[0] if row['C_Identity'] else '100'
        path_A = image_paths.get(A_Identity)
        path_B = image_paths.get(B_Identity)
        path_C = image_paths.get(C_Identity)

        if not path_A or not path_B or not path_C:
            print(f"Missing image path for trial {trial_index}")
            return None

        trial_data = {
            'A_Identity': A_Identity,
            'B_Identity': B_Identity,
            'C_Identity': C_Identity,
            'time_points': []
        }

        with ThreadPoolExecutor(max_workers=4) as executor:  
            future_to_time_point = {}

            for time_point, coordinates_list in trial.items():
                if not isinstance(coordinates_list, list) or len(coordinates_list) != 12:
                    trial_data['time_points'].append({
                        'time_point': time_point,
                        'coords_stimA': None,
                        'coords_stimB': None,
                        'coords_stimC': None,
                        'transformed_outline_A': None,
                        'transformed_outline_B': None,
                        'transformed_outline_C': None,
                    })
                    continue

                coords_stimA = np.array([coord[:2] for coord in coordinates_list[:4]])
                coords_stimB = np.array([coord[:2] for coord in coordinates_list[4:8]])
                coords_stimC = np.array([coord[:2] for coord in coordinates_list[8:12]])    

                if np.isnan(coords_stimA).any() or np.isnan(coords_stimB).any() or np.isnan(coords_stimC).any():
                    trial_data['time_points'].append({
                        'time_point': time_point,
                        'coords_stimA': coords_stimA.tolist(),
                        'coords_stimB': coords_stimB.tolist(),
                        'coords_stimC': coords_stimC.tolist(),
                        'transformed_outline_A': None,
                        'transformed_outline_B': None,
                        'transformed_outline_C': None,
                    })
                    continue

                # Submit tasks
                future_A = executor.submit(process_image_parallel, path_A, coords_stimA)
                future_B = executor.submit(process_image_parallel, path_B, coords_stimB)
                future_C = executor.submit(process_image_parallel, path_C, coords_stimC)

                future_to_time_point[(future_A, future_B, future_C)] = (time_point, coords_stimA, coords_stimB, coords_stimC)

            for futures, (time_point, coords_stimA, coords_stimB, coords_stimC) in future_to_time_point.items():
                transformed_outline_A = futures[0].result()
                transformed_outline_B = futures[1].result()
                transformed_outline_C = futures[2].result() 
                trial_data['time_points'].append({
                    'time_point': time_point,
                    'coords_stimA': coords_stimA.tolist(),
                    'coords_stimB': coords_stimB.tolist(),
                    'coords_stimC': coords_stimC.tolist(),
                    'transformed_outline_A': transformed_outline_A.tolist() if transformed_outline_A is not None else None,
                    'transformed_outline_B': transformed_outline_B.tolist() if transformed_outline_B is not None else None,
                    'transformed_outline_C': transformed_outline_C.tolist() if transformed_outline_C is not None else None,
                })

        return trial_index, trial_data

    # Save data to HDF5 incrementally
    def save_to_hdf5(trial_index, trial_data, hdf5_file):
        with h5py.File(hdf5_file, 'a') as f:
            grp = f.create_group(f"trial_{trial_index}")
            grp.attrs['A_Identity'] = trial_data['A_Identity']
            grp.attrs['B_Identity'] = trial_data['B_Identity']
            grp.attrs['C_Identity'] = trial_data['C_Identity']
            
            for idx, tp_data in enumerate(trial_data['time_points']):
                tp_grp = grp.create_group(f"time_point_{tp_data['time_point']}")
                tp_grp.create_dataset("coords_stimA", data=tp_data['coords_stimA'] or [])
                tp_grp.create_dataset("coords_stimB", data=tp_data['coords_stimB'] or [])
                tp_grp.create_dataset("coords_stimC", data=tp_data['coords_stimC'] or [])
                tp_grp.create_dataset("transformed_outline_A", data=tp_data['transformed_outline_A'] or [])
                tp_grp.create_dataset("transformed_outline_B", data=tp_data['transformed_outline_B'] or [])
                tp_grp.create_dataset("transformed_outline_C", data=tp_data['transformed_outline_C'] or [])


    # Run the trials and save them to HDF5 in batches to limit memory usage
    batch_size = 5  # process 5 trials at a time
    total_trials = len(trials_reordered_list)

    for batch_start in range(0, total_trials, batch_size):
        batch_end = min(batch_start + batch_size, total_trials)
        print(f"Processing batch: trials {batch_start} to {batch_end - 1}")

        with ProcessPoolExecutor(max_workers=2) as executor:
            futures = {executor.submit(process_trial, trial_index, trials_reordered_list[trial_index]): trial_index
                    for trial_index in range(batch_start, batch_end)}

            for future in futures:
                result = future.result()
                if result is not None:
                    trial_index, trial_data = result
                    if trial_data is not None:
                        save_to_hdf5(trial_index, trial_data, output_file)
                del result

        gc.collect()
