import sys
import os
import glob
import shutil
import pathlib as p

import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import spikeinterface.sorters as ss
import spikeinterface.postprocessing as spost
import spikeinterface.qualitymetrics as sqm
import spikeinterface.comparison as sc
import spikeinterface.exporters as sexp
import spikeinterface.widgets as sw

from probeinterface import Probe, get_probe
from probeinterface.plotting import plot_probe
from probeinterface import generate_linear_probe

from preprocessing.preprocessing_utilities import create_target_folder, change_permissions_recursively

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
    #     species = all_folders[0]
    #     project = all_folders[1]
    #     subject = all_folders[2]
    #     date = all_folders[3]
    #     session = all_folders[4]

    return species, project, subject, date, experiment, session

def create_standard_container(project, subject, date, experiment, session, extension):
    return "-".join([project, subject, date, experiment, session]) + extension

as_folder = str(sys.argv[1])

##this generates our current probe from Cambridge Neurotech, since it is not in the list yet
##https://www.cambridgeneurotech.com/assets/files/ASSY-196-H4-map.pdf
linear_probe = generate_linear_probe(num_elec=32, ypitch=25, contact_shapes= 'rect', contact_shape_params= {'width': 11, 'height': 15})
linear_probe.set_contact_ids([18,1,20,3,22,5,23,8,24,7,6,21,4,19,2,17,32,15,30,13,28,11,25,10,26,9,12,27,14,29,16,31])

linear_probe.set_device_channel_indices([45,41,43,38,40,35,36,32,31,33,34,42,37,44,39,46,14,22,17,25,18,28,27,30,23,29,26,21,24,19,20,16])

## Load the nwb file into spikeinterface

species, project, subject, date, experiment, session = get_session_info(as_folder)

cs_folder = create_target_folder('/mnt/hpc_slurm/projects', species, project, subject, date, experiment, session)

species, project, subject, date, experiment, session = get_session_info(cs_folder)
target_file = create_standard_container(project, subject, date, experiment, session, '.nwb')
nwb_filename = os.path.join(cs_folder, target_file)
 # _1 or _2 ?

recording = se.read_nwb(nwb_filename, load_recording=True, load_sorting=False, electrical_series_name='ElectricalSeries_1')
if recording.get_num_channels() == 8:
    print('Only found 8 channels, probably ADC, loading ElectricalSeries_2 instead')
    recording = se.read_nwb(nwb_filename, load_recording=True, load_sorting=False, electrical_series_name='ElectricalSeries_2')
    if recording.get_num_channels() == 8:
            raise ValueError('Both ElectricalSeries have 8 channels, code cant discriminate raw data for %s'%(nwb_filename))

## set the generated probe as the probe used in the recording
recording = recording.set_probe(linear_probe)

## preprocessing recording

recording_cmr = recording
recording_f = spre.bandpass_filter(recording, freq_min=300, freq_max=6000)
print(recording_f)
recording_cmr = spre.common_reference(recording_f, reference='global', operator='median')
print(recording_cmr)

## generate the output folder of the sorting

sorting_folder = os.path.join(cs_folder, 'sorted/')

if not os.path.isdir(sorting_folder):
    os.makedirs(sorting_folder, 0o770, exist_ok=True)
    print("created folder : ", sorting_folder)

preprocessed_folder = os.path.join(cs_folder, 'preprocessed/')

# this computes and saves the recording after applying the preprocessing chain
recording_preprocessed = recording_cmr.save(format='binary', folder = preprocessed_folder, overwrite = True)
print(recording_preprocessed)

## now we spikesort using kiloseort 3

## we add the kilosort repository to the path
ss.Kilosort3Sorter.set_kilosort3_path('/mnt/hpc_slurm/departmentN5/external-repos/Kilosort')

## and sort

sorting = ss.run_sorter(sorter_name = 'kilosort3', recording = recording_preprocessed, output_folder = sorting_folder, verbose = True)

## remove empty units, messes up stuff later

sorting = sorting.remove_empty_units()

## after sorting, we extract waveforms

waveforms_folder = os.path.join(sorting_folder, 'waverforms/')

waveforms = si.extract_waveforms(recording_preprocessed, sorting, folder = waveforms_folder, overwrite=True, use_relative_path = True, sparse=True)

## do postprocessing on the waveforms and compute quality metrics
noise = spost.compute_noise_levels(waveforms)
amplitudes = spost.compute_spike_amplitudes(waveforms)
unit_locations = spost.compute_unit_locations(waveforms)
spike_locations = spost.compute_spike_locations(waveforms)
correlograms, bins = spost.compute_correlograms(waveforms)
similarity = spost.compute_template_similarity(waveforms)

pc = spost.compute_principal_components(waveforms, load_if_exists=True,
                                     n_components=3, mode='by_channel_local')

##qm_params = sqm.get_default_qm_params()

qm_list = [
    'num_spikes',
    'firing_rate',
    'presence_ratio',
    'snr',
    'isi_violation',
    'rp_violation',
    ##'sliding_rp_violation',
 'amplitude_cutoff',
    'amplitude_median',
    'drift'
]

pca_list = [
    ##'isolation_distance',
    ##'l_ratio',
    ##'d_prime',
    'nearest_neighbor',
    ##'nn_isolation',
    ##'nn_noise_overlap'
            ]

qm = sqm.compute_quality_metrics(waveforms, metric_names = qm_list)

pc_metrics = sqm.calculate_pc_metrics(pc, metric_names = pca_list, progress_bar = True)

## export to Phy format for using it with the GUI

phy_folder = os.path.join(sorting_folder, 'phy/')

sexp.export_to_phy(waveforms, phy_folder, verbose=True, remove_if_exists=True, )


change_permissions_recursively(sorting_folder)

