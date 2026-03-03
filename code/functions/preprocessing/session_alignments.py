from preprocessing import align_ephys as align
from preprocessing.preprocessing_utilities import check_recalls, get_session_info
import os
import time
import sys

if len(sys.argv) == 1:
    raise ValueError('Please pass in an archive folder')
else:
    archive_folder = sys.argv[1]

# Check for folder existence
if not os.path.isdir(archive_folder):
    raise ValueError('%s is not a directory'%(archive_folder))
    
# Check if it needs to be recalled
files2recall = check_recalls(archive_folder, recall=False)
if len(files2recall) == 1:
    folder, file = os.path.split(files2recall[0])
    if file == 'sync_messages.txt':
        try:
            alignments = align.snippet_eventmarkers(archive_folder)
        except Exception as ex:
            print(ex)
        
        print('Empty sync_messages.txt, attempting to add the following to file:')
        import numpy as np
        start_ts = np.load(folder + '/continuous/Rhythm_FPGA-100.0/timestamps.npy', mmap_mode='r')[0]
        str_write = 'Software time: %i@10000000Hz\nProcessor: Rhythm FPGA Id: 100 subProcessor: 0 start time: %i@30000Hz\n'%(start_ts,start_ts)
        print(str_write)
        with open(files2recall[0], 'w') as sync_message:
            the_file.write(str_write)

files2recall = check_recalls(archive_folder, recall=True, output_root='/mnt/pns/recalls/')

if len(files2recall) != 0:
    print('Files are being recalled, please wait until the process is finished')
    print(files2recall)

while len(check_recalls(archive_folder, recall=False)) != 0:
    time.sleep(5)

# realign the eventmarkers to logfile
alignments = align.snippet_eventmarkers(archive_folder)
