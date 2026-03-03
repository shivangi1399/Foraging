import logging
import os
import pathlib as p
import shutil
import sys
import time

import deeplabcut
import preprocessing.preprocessing_utilities as utils

config_path_face = "/mnt/hpc_slurm/departmentN5/dlc-models/mouse-face-upgraded-ZeroNoiseLab-2024-01-30/config.yaml"

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

elif utils.is_compute_storage(video_path):
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


deeplabcut.analyze_videos(
    config=config_path_face,
    videos=cs_video_path,
    destfolder=combinedpath,
    save_as_csv=True,
)

deeplabcut.create_labeled_video(
    config=config_path_face, videos=cs_video_path, destfolder=combinedpath
)


utils.change_permissions_recursively(combinedpath)
