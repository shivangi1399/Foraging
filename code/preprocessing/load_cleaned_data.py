# Cut trials from neural data and saving trial info from log files using cut_trials.py
# We can convert all the data into matlab format using cosmos_data_matlab.m
# Use artifact_rejection.m to remove artifacts from LFP data using fieldtrip, at the end convert data to syncopy friendly format
# Load cleaned data back into syncopy usinf load_cleaned_data.py

import os
import h5py
import numpy as np
import syncopy as spy

# Define sessions and base path
sessions = ['20230213'] #['20230202', '20230203', '20230208', '20230209', '20230213', '20230214'] 
base_path = "/cs/projects/MWzeronoise/Analysis/4Shivangi/Datasets/In_matlab"
output_path = "/cs/projects/MWzeronoise/Analysis/4Shivangi/Datasets/neural_data/stimAalign_cut/clean_full_length"

# Loop over each session
for session in sessions:
    os.makedirs(os.path.join(output_path, session), exist_ok=True)
    mat_path = os.path.join(base_path, session, "Data_FT.mat")

    # Load Syncopy AnalogData object
    dct = spy.load_ft_raw(mat_path)
    adata = dct["Data_FT"] 

    # Specify output folder
    output_folder = os.path.join(output_path, session, "Cleaned_lfp_FT")
    adata.save(output_folder)

    print(f"Saved Syncopy AnalogData at {output_path}")

