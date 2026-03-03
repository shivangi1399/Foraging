
import os
import pathlib as p
import re
import numpy as np
import glob
from natsort import natsorted
import time

import logging

from preprocessing.preprocessing_utilities import create_target_folder, get_session_info, create_standard_container

import pynwb

import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.postprocessing as spost
import spikeinterface.preprocessing as spre

from probeinterface import generate_linear_probe

import syncopy as spy

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def is_concatenated_sorting(sorting_folder):
    
    folder_name = os.path.basename(sorting_folder)
    folder_str = str(folder_name)
    concatenated = len(re.findall(r'\b\d{3}\b', folder_str)) > 1
    logging.info(f'Is concatenated sorting: {concatenated}')
    return concatenated


def get_cs_folder_from_nwb_path(nwb_path):
    
    return os.path.dirname(nwb_path)

def get_nwb_files_from_sorting(sorting_folder):

    if not isinstance(sorting_folder, p.Path):
        sorting_folder = p.Path(sorting_folder)
    
    identifier = os.path.basename(sorting_folder).replace('sorted-', '')
    session_identifiers = identifier.split('-')

    base_path = sorting_folder.parent
    logging.info(f"Looking for NWB files in base_path: {base_path}")
    all_nwb_files = list(base_path.rglob('*.nwb'))

    relevant_nwb_files = [file for file in all_nwb_files if any(session_id in file.parts for session_id in session_identifiers)]

    # Return the sorted list of relevant NWB files
    return sorted(relevant_nwb_files, key=lambda path: path.parent.name)


def check_SUA_files_exist(sorting_folder):
    """
    Checks if Syncopy SUA files for the given sorting folder already exist.

    Parameters:
    sorting_folder (str or pathlib.Path): Path to the sorting folder.

    Returns:
    bool: True if Syncopy SUA files exist for all sessions, False otherwise.
    """

    if not isinstance(sorting_folder, p.Path):
        sorting_folder = p.Path(sorting_folder)

    concatenated = is_concatenated_sorting(sorting_folder)
    nwb_paths = get_nwb_files_from_sorting(sorting_folder)

    for nwb_path in nwb_paths:
        cs_folder = get_cs_folder_from_nwb_path(nwb_path)
        spy_container = create_standard_container_from_cs_folder(cs_folder, '.spy')
        spy_container_path = p.Path(cs_folder) / spy_container

        # Check for any files containing '_SUA.spike' in the .spy container
        sua_files_exist = any('_SUA.spike' in file.name for file in spy_container_path.glob('*_SUA.spike.*'))

        if not sua_files_exist:
            logging.info(f"No SUA files found in container: {spy_container_path}")
            return False

    logging.info("All Syncopy SUA files exist.")
    return True

def create_standard_container_from_cs_folder(cs_folder, extension):
    """
    Creates a standard container name based on the cs file path.

    Parameters:
    cs_folder (str or pathlib.Path): Path to the session computer folder.

    Returns:
    str: Standard container name.
    """
    species, project, subject, date, experiment, session = get_session_info(cs_folder)
    return create_standard_container(project, subject, date, experiment, session, extension)


def add_linear_probe_to_recording(recording, nwb_filename):
    # Generate the linear probe
    linear_probe = generate_linear_probe(num_elec=32, ypitch=25, contact_shapes='rect', contact_shape_params={'width': 11, 'height': 15})
    linear_probe.set_contact_ids([18,1,20,3,22,5,23,8,24,7,6,21,4,19,2,17,32,15,30,13,28,11,25,10,26,9,12,27,14,29,16,31])

    # Read channel information from NWB file
    nwbio = pynwb.NWBHDF5IO(nwb_filename, "r", load_namespaces=True)
    nwbfile = nwbio.read()

    channels = nwbfile.electrodes['group_name'][:] == 'CH'
    channel_names = nwbfile.electrodes['location'][:][list(channels)]

    # Determine if channels are mapped and set device channel indices accordingly
    if (natsorted(channel_names) == list(channel_names)):
        print("Channels are not mapped")
        linear_probe.set_device_channel_indices([45,41,43,38,40,35,36,32,31,33,34,42,37,44,39,46,14,22,17,25,18,28,27,30,23,29,26,21,24,19,20,16])
    else:
        print("Channels are mapped")
        linear_probe.set_device_channel_indices(np.arange(32))

    # Set the generated probe as the probe used in the recording
    recording_with_probe = recording.set_probe(linear_probe)
    return recording_with_probe


def load_single_recording(nwb_filename):
    recording = se.read_nwb(nwb_filename, load_recording=True, load_sorting=False, electrical_series_name='ElectricalSeries_1')
    if recording.get_num_channels() == 8:
        recording = se.read_nwb(nwb_filename, load_recording=True, load_sorting=False, electrical_series_name='ElectricalSeries_2')
        if recording.get_num_channels() == 8:
            raise ValueError(f'Both ElectricalSeries have 8 channels, cannot discriminate raw data for {nwb_filename}')

    recording = add_linear_probe_to_recording(recording, nwb_filename)
    
    return recording



def load_nwb_recordings(nwb_paths):

    recordings = [load_single_recording(nwb_file) for nwb_file in nwb_paths]

    return recordings if len(recordings) > 1 else recordings[0]


def process_sorting_segment(recording, sorting_segment, cs_folder):

    logging.info(f'Processing sorting segment in {cs_folder}')

    recording_f = spre.bandpass_filter(recording, freq_min=300, freq_max=6000)
    
    recording_cmr = spre.common_reference(recording_f, reference='global', operator='median')
    
    # Extract waveforms
    output_folder = os.path.join(cs_folder, 'waveforms')
    waveforms_segment = si.extract_waveforms(recording_cmr, sorting_segment, output_folder, overwrite=True, sparse=True, max_spikes_per_unit=200, method='best_channels', num_channels=1)

    # Create SUA file with Syncopy
    create_sua_from_waveforms(sorting_segment, waveforms_segment, cs_folder)




def create_sua_from_waveforms(sorting, waveform, cs_folder):

    if 'sd' in locals():

        sd.clear()

        sd._close()
    
    units = sorting.get_unit_ids()
    all_spikes = sorting.get_all_spike_trains()[0]


    # find channel per unit
    templates = waveform.get_all_templates(mode='median')
    all_max_chan = np.full(units.max()+1, -1)
    all_max_chan[units] = templates.max(axis=1).argmax(axis=1)

    print('Creating and saving sua spy')
    species, project, subject, date, experiment, session = get_session_info(cs_folder)
    target_file = create_standard_container(project, subject, date, experiment, session, '.spy')
    spy_filename = os.path.join(cs_folder, target_file)

    sd = spy.SpikeData(data = np.column_stack([all_spikes[0], all_max_chan[all_spikes[1]], all_spikes[1]]).astype(int), 
                       dimord = ['sample', 'channel', 'unit'], 
                       samplerate = sorting.sampling_frequency)
    
    if 'mua' in locals():

        mua.clear()
    
        mua._close()

    mua = spy.load(spy_filename, tag = 'spikes')
    sd.info = mua.info

    spy.save(sd, container = spy_filename, tag = 'SUA', overwrite=True)

    return sd




def process_curated_sorting(sorting_folder, to_exclude = ["noise","mua"], overwrite = False):

    if not isinstance(sorting_folder, p.Path):
        sorting_folder = p.Path(sorting_folder)

        # Check if SUA files exist and handle overwrite option
    if not overwrite and check_SUA_files_exist(sorting_folder):
        logging.info("SUA files already exist. Set overwrite=True to regenerate them.")
        return
        
    concatenated = is_concatenated_sorting(sorting_folder)

    sorting_curated = se.read_phy(os.path.join(sorting_folder, "phy/"), exclude_cluster_groups = to_exclude)

    if concatenated:
        logging.info('Processing concatenated sorting')
        
        nwb_paths = get_nwb_files_from_sorting(sorting_folder)

        time.sleep(5)
        
        recording_list = load_nwb_recordings(nwb_paths)
        sorting_list = si.split_sorting(sorting_curated, recording_list)

        for i in range(len(recording_list)):
            recording = recording_list[i]
            sorting_segment = si.select_segment_sorting(sorting_list, [i])
            cs_folder = get_cs_folder_from_nwb_path(nwb_paths[i])

            time.sleep(5)
            
            process_sorting_segment(recording, sorting_segment, cs_folder)
    else:
        logging.info('Processing single-session sorting')
        
        nwb_paths = get_nwb_files_from_sorting(sorting_folder)

        time.sleep(5)

        recording = load_nwb_recordings(nwb_paths)
        cs_folder = get_cs_folder_from_nwb_path(nwb_paths[0])

        time.sleep(5)

        process_sorting_segment(recording, sorting_curated, cs_folder)
