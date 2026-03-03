import os
import shutil
from datetime import datetime
import getpass

# file processing helper
def path_split(path):
    path = os.path.normpath(path)
    return path.split(os.sep)

def get_session_info(path):
    all_folders = path_split(path)

    if all_folders[0].endswith(':'):
        start_folders = get_windows_mount_location(all_folders.pop(0))
        all_folders = start_folders + all_folders

    while all_folders.pop(0) != 'projects':
        continue
    
    species, project, subject, date, experiment, session = all_folders

    return species, project, subject, date, experiment, session

def get_windows_mount_location(dl):
    import win32file
    # ['', 'Device', 'LanmanRedirector', ';W:000000001403c7b8', 'as', 'projects', 'OWzeronoise']

    flds = win32file.QueryDosDevice(dl).split("\x00")[0].split(os.sep)
    return flds[4:]

def create_target_folder(target_root, species, project, subject, date, experiment, session):
    all_folders = path_split(target_root)
    if all_folders[0].endswith(':'):
        mount_drive = all_folders[0]
        start_folders = get_windows_mount_location(all_folders.pop(0))
        all_folders = start_folders + all_folders
    
    if all_folders[-1] == 'projects':
        target_folder = os.path.join(target_root, species, project, subject, date, experiment, session)
        return target_folder
    elif all_folders[-1] == species or all_folders[-1].startswith(('ZeroNoise_', 'MWzeronoise', 'OWzeronoise')):
        # already have a species identifier
        target_folder = os.path.join(target_root, project, subject, date, experiment, session)
        return target_folder
    elif all_folders[-1].startswith(('cs','gs','as','hpx','pns','archive')):
        # missing projects
        target_folder = os.path.join(target_root, 'projects', species, project, subject, date, experiment, session)
        return target_folder
    else:
        # try to figure out how far along the path we are
        if all_folders[-1] == session:
            target_folder = target_root
            return target_folder

        possible_folders = [project, subject, date, session]
        for ii, fold in possible_folders:
            if all_folders[-1] == fold:
                target_folder = os.path.join(target_root, possible_folders[ii+1:])
                return target_folder
        
        raise ValueError('target_root does not have matching folder structure, specify target_fullfile instead')

def create_standard_container(project, subject, date, experiment, session, extension):
    return "-".join([project, subject, date, experiment, session]) + extension

def archive_recall(files2recall, output_root):
    recall_folder = '/mnt/archive/recalls/'
    temp_folder = '/tmp/'
    
    print('Recall time')
    now = datetime.now().strftime('%Y-%m-%d_%H-%M')
    user = getpass.getuser()
    recall_name = '%s_%s.rcl'%(now, user)

    with open(temp_folder+recall_name, "w") as recall:
        for ifile in files2recall:
            recall.write('%s\n'%(ifile))
            
    output_folder = output_root+recall_name[:-4]
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
    else: 
        raise ValueError('%s already exists, try again in 1 minute',output_folder)
    
    with open(recall_folder+recall_name+'.out', "w") as recalldir:
        recalldir.write('%s\n'%(output_folder))
        
    print('Recalling %i files, expect an email'%(len(files2recall)))
    shutil.move(temp_folder+recall_name, recall_folder)


def check_recalls(archive_folder, recall=True, output_root='/mnt/pns/recalls/'):
    files2recall = []

    if os.path.isfile(archive_folder):
        # Directly check the file without needing cur_path or ifile
        if os.stat(archive_folder).st_blocks == 0:
            files2recall.append(archive_folder)
    else:
        # Iterate over the files in the directory
        for cur_path, _, files in os.walk(archive_folder):
            for ifile in files:
                file_path = os.path.join(cur_path, ifile)
                if os.stat(file_path).st_blocks == 0:
                    files2recall.append(file_path)

    # Recall the files or report the number of files to be recalled
    if len(files2recall) > 0:
        if recall:
            archive_recall(files2recall, output_root)
        else:
            print(len(files2recall), 'files need recalling')

    return files2recall
import os

def change_permissions_recursively(root_path, desired_perm=0o2770):
    try:
        current_user_uid = os.getuid()  # Get the current user's UID
        
        if os.path.isfile(root_path):
            # If root_path is a file, check ownership and change its permissions if owned by the user
            if is_owned_by_user(root_path, current_user_uid):
                update_permissions_if_needed(root_path, desired_perm)
        else:
            # If root_path is a directory, proceed with the original logic
            for foldername, subfolders, filenames in os.walk(root_path):
                # Check ownership and change the permission of the current directory if owned by the user
                if is_owned_by_user(foldername, current_user_uid):
                    update_permissions_if_needed(foldername, desired_perm)
                
                # Iterate through subdirectories and files, checking ownership and changing permissions if necessary
                for subfolder in subfolders:
                    subfolder_path = os.path.join(foldername, subfolder)
                    if is_owned_by_user(subfolder_path, current_user_uid):
                        update_permissions_if_needed(subfolder_path, desired_perm)
                
                for filename in filenames:
                    file_path = os.path.join(foldername, filename)
                    if is_owned_by_user(file_path, current_user_uid):
                        update_permissions_if_needed(file_path, desired_perm)
        
        print(f"Permissions updated to {oct(desired_perm)} where necessary for all items in {root_path}")

    except Exception as e:
        print(f"Unable to change permissions recursively for {root_path}. Error: {e}")

def update_permissions_if_needed(path, desired_perm):
    current_perm = os.stat(path).st_mode & 0o7777
    if current_perm != desired_perm:
        os.chmod(path, desired_perm)

def is_owned_by_user(path, user_uid):
    """Check if the given path is owned by the user with the specified UID."""
    return os.stat(path).st_uid == user_uid

def is_compute_storage(path):
    """
    Checks if the path belongs to compute storage.
    """
    compute_prefixes = ['/mnt/hpc/', '/mnt/hpc_slurm/', '/hpc/', '/hpc_slurm/', '/cs/', '/mnt/cs/']
    return any(path.startswith(prefix) for prefix in compute_prefixes)



def is_archive_storage(path):
    """
    Checks if the path belongs to compute storage.
    """
    archive_prefixes = ['/mnt/as/', '/as/']
    return any(path.startswith(prefix) for prefix in archive_prefixes)

def is_video_file(path):
    """
    Checks if the path is a video file.
    """
    video_extensions = ['.mp4', '.avi', '.mov', '.mpeg', '.mpg', '.wmv']
    return any(path.endswith(ext) for ext in video_extensions)
