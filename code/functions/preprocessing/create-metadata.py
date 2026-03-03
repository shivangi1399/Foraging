import numpy as np
import pandas as pd
import re
import glob
import os
import sys

from parse_logfile import TextLog
from preprocessing.align_flashes import extract_flashes
from preprocessing.preprocessing_utilities import check_recalls

if len(sys.argv) == 1:
    raise ValueError('Please pass in an archive folder')
else:
    archive_path = sys.argv[1]


# Check for folder existence
if not os.path.isdir(archive_path):
    raise ValueError('%s is not a directory'%(archive_path))

    
if any([x in archive_path for x in ['OWzeronoise', 'ZeroNoise_MOUSE']]):
    meta_file_path = '/mnt/hpc_slurm/departmentN5/behaviour/mouse/meta-mouse.csv'
if any([x in archive_path for x in ['MWzeronoise', 'ZeroNoise_MONKEY']]):
    meta_file_path = '/mnt/hpc_slurm/departmentN5/behaviour/monkey/meta-monkey.csv'

files2recall = check_recalls(archive_path, recall=True, output_root='/mnt/pns/recalls/')
if len(files2recall) != 0:
    print('Files are being recalled, please wait until the process is finished and then try again')
    print(files2recall)


logfile_paths = list(set(glob.glob(archive_path+"/*Cont.log")) - set(glob.glob(archive_path+"/*Start*.log")))

video_paths = glob.glob(archive_path + "/*.avi")

# Repeat everything, for the mouse.
list_tmp = ['Subject', 'Selected State Machine', 'Experiment', 'Session']
data_list = []

print("Extracting session info.")

for ipath, path in enumerate(logfile_paths):

    with TextLog(path) as log:
        starts = log.count_coincidences(search_string = 'TransitionPassed,StartTrial')
    ##    if starts > 50:
        corrects = log.count_coincidences(search_string = 'TransitionPassed,Correct')
        info = log.read_log_header_with_return()
        if 'Selected State Machine' in list(info.keys()):
            date = re.search('[0-9]{4}\_[0-9]{2}\_[0-9]{2}', path).group()
            useful_info = np.concatenate([[info[x] for x in list_tmp], [starts], [corrects/starts], [date], [path]])
            data_list.append(useful_info)
    list_tmp.append('Number of trials')
    list_tmp.append('Performance')
    list_tmp.append('Date')
    list_tmp.append('LogFile Path')

    if ipath < 2:
        df = pd.DataFrame(data_list, columns = list_tmp)
    else:
        df.append(pd.DataFrame(data_list, columns = list_tmp))

df[['Subject', 'Selected State Machine', 'Experiment', 'Session', 'Date', 'LogFile Path']] = df[['Subject', 'Selected State Machine', 'Experiment', 'Session', 'Date', 'LogFile Path']].astype('string')

df[['Performance']] = df[['Performance']].astype(np.float64)

df[['Number of trials']] = df[['Number of trials']].astype(np.int64)

if os.path.exists(meta_file_path):
        meta_file = pd.read_csv(meta_file_path,
                            dtype = {'Subject' : 'string',
                                     'Selected State Machine' : 'string',
                                     'Experiment' : 'string',
                                     'Session' : 'string',
                                     'Date' : 'string',
                                     'LogFile Path' : 'string',
                                     'Performance' : np.float64,
                                     'Number of trials' : np.int64,
                                     'flash1' : np.int64,
                                     'flash2' : np.int64,
                                     'flash3' : np.int64})

        if len(df.merge(meta_file)) == len(df):
            print("already ran, skipping")
            exit()
        
flash_nums = np.empty(3) * np.nan
        
for ipath,path in enumerate(video_paths):
    print("Extracting flashes from video:")
    print(path)
    diffs = extract_flashes(path, pixel_jump=100)[2]
    thresholds = np.arange(0.1,0.5, 0.01)
    sums = []
    for idx, thresh in enumerate(thresholds):
        sums.append(np.sum(diffs > thresh))
    print(np.median(sums))
    flash_nums[ipath] = np.median(sums)

df['flash1'], df['flash2'], df['flash3'] = flash_nums

df['LogFile Path'] = df.pop('LogFile Path')

df[['flash1', 'flash2', 'flash3']] = df[['flash1', 'flash2', 'flash3']].astype(np.int64)

if not os.path.exists(meta_file_path):
    df.to_csv(meta_file_path, index = False, )
else: 
    meta_file = pd.read_csv(meta_file_path,
                            dtype = {'Subject' : 'string',
                                     'Selected State Machine' : 'string',
                                     'Experiment' : 'string',
                                     'Session' : 'string',
                                     'Date' : 'string',
                                     'LogFile Path' : 'string',
                                     'Performance' : np.float64,
                                     'Number of trials' : np.int64,
                                     'flash1' : np.int64,
                                     'flash2' : np.int64,
                                     'flash3' : np.int64})
    
    if len(df.merge(meta_file)) == len(df):
        print("already ran, skipping.")
    else:
        df.to_csv(meta_file_path, mode='a', header=False, index = False)

    
