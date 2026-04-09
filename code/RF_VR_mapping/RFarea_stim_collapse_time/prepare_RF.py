# RF coordinates in cartesian coordinates

"""
#### Code instructions #######################################################################################################
# 1. run mapping_corner.py for stimuli corner in cartesian coordinates
# 2. run prepare_RF.py for Rfs in cartseian coordinates
# 3. order the corner using corner_order.py and run warping_stim.py to warp the stimuli in the corners we got in step 1
# 4. Check if RF falls inside the warped stim accounting for collapse using RFarea_stim_collapse.py
# 5. Check how everything looks using RFoverlap_perc_collapse.py
##############################################################################################################################
"""

import os
import sys
import pickle
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
from scipy import interpolate          
import h5py
import matplotlib.pyplot as plt
from skimage.transform import ProjectiveTransform
from skimage import io
from skimage.filters import threshold_otsu
from skimage.measure import find_contours
from shapely.geometry import Polygon, Point
sys.path.insert(1, '/mnt/cs/projects/MWzeronoise/Analysis/4Shivangi/code/functions/unreal_logfile')
from parse_logfile import TextLog #run this in warping env


sessions = ['20230202', '20230206', '20230209'] #list of session whaich have bar mapping data

for session in sessions:
    os.chdir(f'/mnt/hpc/projects/MWzeronoise/Analysis/4Shivangi/Results/RF_VR_mapping/RFs/gaussian_fits/{session}/')
    Fit_az_ele = pd.read_csv("Fit_az_ele.csv", header=None).to_numpy(dtype=float) #obtained by running gaussian fits on the RFs in 
                    #/mnt/hpc/projects/MWzeronoise/Analysis/4Shivangi/code/RF_VR_mapping/prep_code/RF_gaussian_fit.m

    # Extract the centers and the radius
    last_three_columns = Fit_az_ele[:, -3:]
    RFs = np.round(last_three_columns, 3)

    elevation = RFs[:, 0]
    azimuth = RFs[:, 1]
    radius_spherical = RFs[:, 2]

    def sphere2cartesian(azimuth, elevation, R=1):
        x = R*np.sin((np.deg2rad(elevation)))*np.cos(np.deg2rad(azimuth))
        y = R*np.sin(np.deg2rad(elevation))*np.sin(np.deg2rad(azimuth))
        z = R*np.cos(np.deg2rad(elevation))
        return np.column_stack([x,y,z])

    # radius and center
    cartesian_coords = sphere2cartesian(azimuth, elevation) # spherical to retinal to cartesian conversion makes no difference
    az_r = azimuth + radius_spherical + 1 #increase the radius by 1 degree
    el_r = elevation
    cartesian_coords_r = sphere2cartesian(az_r, el_r)
    distances = np.linalg.norm(cartesian_coords - cartesian_coords_r, axis=1)

    outdir = f'/cs/projects/MWzeronoise/Analysis/4Shivangi/Results/RF_VR_mapping/RFs/center_radius/{session}/'
    os.makedirs(outdir, exist_ok=True)
    os.chdir(outdir)
    np.save('center_coords.npy', cartesian_coords)
    np.save('RF_radius.npy', distances)