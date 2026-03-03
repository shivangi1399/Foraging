import syncopy as spy
from preprocessing import align_ephys as align
from preprocessing import snippet_ephys as snip
from preprocessing import preprocess_ephys as prep
from preprocessing.preprocessing_utilities import check_recalls
from parse_logfile import TextLog
import os
import glob
import sys
import subprocess

from syncopy import __storage__
import time


if len(sys.argv) == 1:
    print('No file name given, please enter manually and I will construct the filename.')
    subj = input("Subject name:\n")
    #print(f'You entered {subj}')
    if subj[0].isnumeric():
        root = '/as/projects/OWzeronoise/*/%s'%(subj)
    else:
        root = '/as/projects/MWzeronoise/*/%s'%(subj)
    rec = input("Recording date:\n")
    #print(f'You entered {rec}')

    # get list of files that matches pattern
    pattern = '%s/%s/*/*'%(root, rec)
    dirs = list(filter(os.path.isdir, glob.glob(pattern)))

    for ii, this_dir in enumerate(dirs):
        print(ii, this_dir)
    idx = input("Type the number of the archive folder to analyse:\n")

    print('You selected: ', dirs[int(idx)])
    ans = input("Is this correct? y/n:\n")
    if ans != 'y':
        sys.exit("Wrong archive folder give, press ctrl c and try again")
    archive_folder = dirs[int(idx)]

else:
    archive_folder = sys.argv[1]

# Check for folder existence
if not os.path.isdir(archive_folder):
    raise ValueError('%s is not a directory'%(archive_folder))

# Check if it needs to be recalled

files2recall = check_recalls(archive_folder, recall=True, output_root='/mnt/pns/recalls/')
if len(files2recall) != 0:
    print('Files are being recalled, please wait until the process is finished and then try again')
    print(files2recall)
            
    while len(check_recalls(archive_folder, recall=False)) != 0:
        time.sleep(5)

files = glob.glob(__storage__+'/*', recursive=True)
for file in files:
    flagDelete = False
    fileAge = (time.time() - os.path.getmtime(file))/3600
    ext = os.path.splitext(file)[-1].lower()
    # Check if h5 and older than a day, delete.
    if (ext == ".h5") * (fileAge > 24):
        flagDelete = True
    # If very big, and older than 6 hours delete.
    elif (os.path.getsize(file) > 50 * 1024 ** 3) and (fileAge > 6):
        flagDelete = True
    if flagDelete:
        print(file+' will be removed')
        os.remove(file)

# convert the data
nwb_filename = prep.convert2nwb(archive_folder)
out_dict = spy.load_nwb(nwb_filename)
raw_data = prep.find_spy_raw_data(out_dict)
if raw_data is None:
    print('No raw data found')
    sys.exit(0)

# create container
spy_filename = nwb_filename[:-4] + '.spy'
if not os.path.isdir(spy_filename):
    original_umask = os.umask(0)
    os.makedirs(spy_filename, mode=2770, exist_ok=True)
    os.umask(original_umask)

# create lfp, eMUA and mua
try:
    files = glob.glob(os.path.join(spy_filename, '*_lfp.analog'))
    if len(files) == 0:
        prep.create_lfp(spy_filename, raw_data=raw_data, n_jobs_startup=20, partition="8GBS", timeout=400)
except Exception as ex:
    print(ex)
    spy.cluster_cleanup()
    try:
        print('Second try with more memory')
        prep.create_lfp(spy_filename, raw_data=raw_data, n_jobs_startup=10, partition="16GBS", timeout=400)
    except Exception as ex:
        print(ex)
        spy.cluster_cleanup()


try:
    files = glob.glob(os.path.join(spy_filename, '*_eMUA.analog'))
    if len(files) == 0:
        prep.create_eMUA(spy_filename, raw_data=raw_data, n_jobs_startup=20, partition="8GBS", timeout=400)
except Exception as ex:
    print(ex)
    spy.cluster_cleanup()
    try:
        print('Second try with more memory')
        prep.create_eMUA(spy_filename, raw_data=raw_data, n_jobs_startup=10, partition="16GBS", timeout=400)
    except Exception as ex:
        print(ex)
        spy.cluster_cleanup()

try:
    files = glob.glob(os.path.join(spy_filename, '*_spikes.spike'))
    if len(files) == 0:
        prep.create_spikes(spy_filename, raw_data=raw_data, n_jobs_startup=20, partition="16GBS", timeout=400)
except Exception as ex:
    print(ex)
    spy.cluster_cleanup()

os.remove(raw_data.filename)

