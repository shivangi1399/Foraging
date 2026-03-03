import glob
import logging
import os
import pathlib as p
import shutil
import sys
import time

import deeplabcut
import numpy as np
import pandas as pd
import preprocessing.preprocessing_utilities as utils
from preprocessing.motion_correction import correct_motion

config_path_pupil = "/mnt/hpc_slurm/departmentN5/dlc-models/mouse-pupil-ZeroNoiseLab-2022-05-02/config.yaml"


video_path = str(sys.argv[1])


if not utils.is_video_file(video_path):
    logging.error("The provided path is not a video file.")
    sys.exit(1)

if utils.is_archive_storage(video_path):
    species, project, subject, date, experiment, session = utils.get_session_info(
        os.path.dirname(video_path)
    )

    outputdir = utils.create_target_folder(
        "/mnt/hpc_slurm/projects", species, project, subject, date, experiment, session
    )

    correct_video_path = os.path.join(
        utils.create_target_folder(
            "/mnt/as/projects", species, project, subject, date, experiment, session
        ),
        os.path.basename(video_path),
    )

    CHECK_FOLDER = os.path.isdir(outputdir)

    # If folder doesn't exist, then create it.
    if not CHECK_FOLDER:
        os.makedirs(outputdir, 0o770, exist_ok=True)
        print("created folder : ", outputdir)

    containername = utils.create_standard_container(
        project, subject, date, experiment, session, extension=".dlc"
    )

    combinedpath = os.path.join(outputdir, containername)

    if not os.path.isdir(combinedpath):
        os.makedirs(combinedpath, 0o770, exist_ok=True)
        print("created folder : ", containername)

    putPath = p.Path(correct_video_path.replace("/mnt/as/", "/mnt/hpc_slurm/"))

    if not putPath.exists():
        files2recall = utils.check_recalls(
            video_path, recall=True, output_root="/mnt/pns/recalls/"
        )
        if len(files2recall) != 0:
            print(
                "Files are being recalled, please wait until the process is finished and then try again"
            )
            print(files2recall)

        while len(utils.check_recalls(video_path, recall=False)) != 0:
            time.sleep(5)

        print("Copying video file to compute storage")

        cs_video_path = shutil.copyfile(correct_video_path, putPath)

    elif putPath.stat().st_size == 0:
        files2recall = utils.check_recalls(
            video_path, recall=True, output_root="/mnt/pns/recalls/"
        )
        if len(files2recall) != 0:
            print(
                "Files are being recalled, please wait until the process is finished and then try again"
            )
            print(files2recall)

        while len(utils.check_recalls(video_path, recall=False)) != 0:
            time.sleep(5)

        print("Copying video file to compute storage")
        cs_video_path = shutil.copyfile(correct_video_path, putPath)

    else:
        cs_video_path = str(putPath)

    utils.change_permissions_recursively(cs_video_path)

elif is_compute_storage(video_path):
    if "/hpc/" in video_path or "/cs/" in video_path:
        video_path = video_path.replace("/hpc/", "/hpc_slurm/").replace(
            "/cs/", "/hpc_slurm/"
        )
    outputdir = os.path.dirname(video_path)
    species, project, subject, date, experiment, session = utils.get_session_info(
        os.path.dirname(video_path)
    )
    containername = utils.create_standard_container(
        project, subject, date, experiment, session, extension=".dlc"
    )
    combinedpath = os.path.join(outputdir, containername)


else:
    logging.error(
        "The provided path does not belong to neither archive not compute. We don't slurm on /gs/!"
    )
    sys.exit(1)


def get_eye_coords(dlc_face_path, buffer_ratio=0.2):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(dlc_face_path, skiprows=[0], header=[0, 1], index_col=0)
    df.columns = df.columns.map("_".join)

    # Function to process each eye coordinate
    def process_eye_coord(df, coord_x, coord_y, likelihood_threshold=0.9):
        x = df[[coord_x, f'{coord_x.split("_")[0]}_likelihood']]
        x = x.loc[x[f'{coord_x.split("_")[0]}_likelihood'] > likelihood_threshold]
        x_median = np.round(np.median(x[coord_x])).astype(int)

        y = df[[coord_y, f'{coord_y.split("_")[0]}_likelihood']]
        y = y.loc[y[f'{coord_y.split("_")[0]}_likelihood'] > likelihood_threshold]
        y_median = np.round(np.median(y[coord_y])).astype(int)

        return x_median, y_median

    # Calculate coordinates for each eye part
    front_x, top_y = process_eye_coord(df, "eyeFront_x", "eyeTop_y")
    back_x, bottom_y = process_eye_coord(df, "eyeBack_x", "eyeBottom_y")

    # Calculate buffer for each coordinate
    width_buffer = int((back_x - front_x) * buffer_ratio)
    height_buffer = int((bottom_y - top_y) * buffer_ratio)

    # Apply buffer to coordinates and ensure non-negative values
    front_x = max(front_x - width_buffer, 0)
    back_x = max(back_x + width_buffer, 0)
    top_y = max(top_y - height_buffer, 0)
    bottom_y = max(bottom_y + height_buffer, 0)

    return [front_x, back_x, top_y, bottom_y]


bare_video_path = os.path.splitext(os.path.basename(cs_video_path))[0]

coords = get_eye_coords(
    glob.glob(combinedpath + "/*" + bare_video_path + "*mouse-face*upgraded*.csv")[0]
)

moco_path = os.path.join(combinedpath, bare_video_path) + "-moco.mp4"

motion_corrected_path = correct_motion(cs_video_path, cropping_coords=coords)

deeplabcut.analyze_videos(
    config=config_path_pupil,
    videos=motion_corrected_path,
    ##                          cropping=coords,
    destfolder=combinedpath,
    save_as_csv=True,
)

deeplabcut.create_labeled_video(
    config=config_path_pupil,
    videos=motion_corrected_path,
    ##save_frames=True,
    destfolder=combinedpath,
    ##displaycropped=True
)


utils.change_permissions_recursively(combinedpath)
