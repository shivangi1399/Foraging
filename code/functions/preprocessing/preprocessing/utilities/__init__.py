from .sua_from_sorting import process_curated_sorting as _process_curated_sorting

def sua_from_sorting(sorting_folder, to_exclude=["noise", "mua"], overwrite=False):
    """
    Wrapper for process_curated_sorting to process sorting data and generate SUA objects.

    Processes a curated sorting to generate Syncopy Single Unit Activity (SUA) files.

    This function takes a sorting folder containing curated spike sorting data and processes it
    to generate SUA files in Syncopy format. It supports processing both concatenated and 
    single-session sortings.

    Parameters:
    sorting_folder (str or pathlib.Path): Path to the sorting folder containing curated data.
    to_exclude (list, optional): List of cluster groups to exclude from processing. 
                                 Defaults to ["noise", "mua"].
    overwrite (bool, optional): If set to True, existing SUA files will be overwritten. 
                                If False, the function will skip processing and exit if SUA files 
                                already exist. Defaults to False.

    Returns:
    None: The function creates SUA files in the appropriate directories and does not return a value.

    If SUA files already exist and overwrite is False, a log message will be displayed to inform 
    the user. To regenerate SUA files, set overwrite to True.

    Raises:
    FileNotFoundError: If the specified sorting_folder does not exist or required files are missing.
    ValueError: If conditions in the NWB files are not met (e.g., both ElectricalSeries with 8 channels).
    ValueError: If the sorting_folder path does not contain the required substrings.

    Example:
    >>> sua = process_curated_sorting('/path/to/sorted-011/', to_exclude=["mua"])

    Note:
    This function handles both concatenated and single-session sorting scenarios.
    """

    # Ensure sorting_folder is a string
    folder_str = str(sorting_folder)

    # Check for required substrings in the path
    if not any(substring in folder_str for substring in ['/hpc/', '/hpc_slurm/', '/cs/']):
        raise ValueError("Path must point to a compute folder and contain one of '/hpc/', '/hpc_slurm/', or '/cs/'")

    # Prepend '/mnt/' if missing
    if not folder_str.startswith('/mnt/'):
        folder_str = '/mnt' + folder_str

    return _process_curated_sorting(folder_str, to_exclude, overwrite)
