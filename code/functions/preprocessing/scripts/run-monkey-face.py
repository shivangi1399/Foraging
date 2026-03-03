import os
import pathlib as p
import shutil
import sys

import deeplabcut
from preprocessing.preprocessing_utilities import create_target_folder

config_path_face = "/mnt/hpc_slurm/departmentN5/dlc-models/monkey-face-ZeroNoiseLab-2022-05-02/config.yaml"

as_video_path = str(sys.argv[1])


# file processing helper
def path_split(path):
    path = os.path.normpath(path)
    return path.split(os.sep)


def get_session_info(path):
    all_folders = path_split(path)

    if all_folders[0].endswith(":"):
        start_folders = get_windows_mount_location(all_folders.pop(0))
        all_folders = start_folders + all_folders

    while all_folders.pop(0) != "projects":
        continue

    species, project, subject, date, experiment, session = all_folders
    #     species = all_folders[0]
    #     project = all_folders[1]
    #     subject = all_folders[2]
    #     date = all_folders[3]
    #     session = all_folders[4]

    return species, project, subject, date, experiment, session


def create_standard_container(project, subject, date, experiment, session, extension):
    return "-".join([project, subject, date, experiment, session]) + extension


species, project, subject, date, experiment, session = get_session_info(
    os.path.dirname(as_video_path)
)

outputdir = create_target_folder(
    "/mnt/hpc_slurm/projects", species, project, subject, date, experiment, session
)

correct_video_path = os.path.join(
    create_target_folder(
        "/mnt/as/projects", species, project, subject, date, experiment, session
    ),
    os.path.basename(as_video_path),
)

CHECK_FOLDER = os.path.isdir(outputdir)

# If folder doesn't exist, then create it.
if not CHECK_FOLDER:
    os.makedirs(outputdir, 0o770, exist_ok=True)
    print("created folder : ", outputdir)

containername = create_standard_container(
    project, subject, date, experiment, session, extension=".dlc"
)

combinedpath = os.path.join(outputdir, containername)

if not os.path.isdir(combinedpath):
    os.makedirs(combinedpath, 0o770, exist_ok=True)
    print("created folder : ", containername)

putPath = p.Path(correct_video_path.replace("/as/", "/hpc_slurm/"))

if not putPath.exists():
    print("Copying video file to compute storage")
    cs_video_path = shutil.copyfile(
        correct_video_path, correct_video_path.replace("/as", "/hpc_slurm/")
    )
elif putPath.stat().st_size == 0:
    print("Copying video file to compute storage")
    cs_video_path = shutil.copyfile(
        correct_video_path, correct_video_path.replace("/as", "/hpc_slurm/")
    )
else:
    cs_video_path = correct_video_path.replace("/as", "/hpc_slurm/")

try:
    os.chmod(cs_video_path, mode=0o770)
except:
    print("Unable to change permissions of files automatically")

deeplabcut.analyze_videos(
    config=config_path_face,
    videos=cs_video_path,
    destfolder=combinedpath,
    save_as_csv=True,
)

deeplabcut.create_labeled_video(
    config=config_path_face, videos=cs_video_path, destfolder=combinedpath
)
try:
    os.chmod(combinedpath, mode=0o770)
except:
    print("Unable to change permissions of folder automatically")
