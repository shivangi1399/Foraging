import os
import numpy as np

from spikeinterface import extract_waveforms, load_extractor, WaveformExtractor
import spikeinterface.extractors as se
from spikeinterface.postprocessing import compute_spike_amplitudes
import  syncopy as spy

from .preprocessing_utilities import create_target_folder, get_session_info, create_standard_container, change_permissions_recursively

def create_spikeinterface_waveforms(recording, sorting, sua_folder, ms_before=2, ms_after=3, best_channels=3, n_jobs=1):
    if sua_folder == None:
        mode = "memory"
    else:
        mode = "folder"
    if best_channels != None:
        sparse_kwargs = dict(sparse = True, method = 'best_channels', num_channels = best_channels)
    else:
        sparse_kwargs = dict(sparse = False)
    we = extract_waveforms(recording, sorting, folder=sua_folder, mode=mode, precompute_template=None,ms_before=ms_before, ms_after=ms_after, max_spikes_per_unit=None, overwrite=True, return_scaled=True, n_jobs=n_jobs, **sparse_kwargs)
    return we

def create_spikeinterface_waveforms(recording, sorting, sua_folder, ms_before=2, ms_after=3, best_channels=3, n_jobs=1):
    if sua_folder == None:
        mode = "memory"
    else:
        mode = "folder"

    if best_channels != None:
        sparse_kwargs = dict(sparse = True, method = 'best_channels', num_channels = best_channels)
    else:
        sparse_kwargs = dict(sparse = False)


    we = extract_waveforms(recording, sorting, folder=sua_folder, mode=mode, precompute_template=None,
                          ms_before=ms_before, ms_after=ms_after, max_spikes_per_unit=None, 
                          overwrite=True, return_scaled=True, 
                          n_jobs=n_jobs, chunk_durations='1s',
                          **sparse_kwargs)
    if sua_folder is not None:
        
        change_permissions_recursively(sua_folder)
        
    
    return we

def load_recording_nwb(cs_folder):
    #nwb
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
    
    return recording


def sorted_to_spy(cs_folder, include_waveform = True, mode = "memory"):
    # find the sorted folder
    si_folder = os.path.join(cs_folder, 'sorted')
    if not os.path.isdir(si_folder):
        raise ValueError('folder given does not contain SUA')
    curated_data = os.path.join(si_folder, 'curated_sorting.pnz.npz')
    if not os.path.isfile(curated_data):
      raise ValueError('folder given does not contain curated SUA')

    print('Loading curated sua')

    #sorting = se.read_kilosort(si_folder, keep_good_only=True)
    sorting = se.NpzSortingExtractor(curated_data)
    units = sorting.get_unit_ids()
    all_spikes = sorting.get_all_spike_trains()[0]
    
    
    # check for pre-existing waveforms
    waveform_folder = os.path.join(si_folder, 'waveform')
    if not os.path.isdir(waveform_folder):
        prep_folder = os.path.join(si_folder, 'preprocessed')
        recording = load_extractor(prep_folder)
        #recording = load_recording(cs_folder)

        # if no waveforms then create:
        if mode == "memory":
            waveform_folder = None
            
        waveform = create_spikeinterface_waveforms(recording, sorting, waveform_folder)
    else:
        waveform = WaveformExtractor.load(waveform_folder)
    
    #waveform.precompute_templates(modes=['median'])
    # find channel per unit
    templates = waveform.get_all_templates(mode='median')
    all_max_chan = np.full(units.max()+1, -1)
    all_max_chan[units] = templates.max(axis=1).argmax(axis=1)
    
    #amps = compute_spike_amplitudes(waveform, load_if_exists=True, peak_sign='both', return_scaled=True, outputs='concatenated', **job_kwargs)

    print('Creating and saving sua spy')
    species, project, subject, date, experiment, session = get_session_info(cs_folder)
    target_file = create_standard_container(project, subject, date, experiment, session, '.spy')
    spy_filename = os.path.join(cs_folder, target_file)

    sd = spy.SpikeData(data = np.column_stack([all_spikes[0], all_max_chan[all_spikes[1]], all_spikes[1]]).astype(int), 
                       dimord = ['sample', 'channel', 'unit'], 
                       samplerate = sorting.sampling_frequency)
    

    if include_waveform:
        # save waveform attribute of spy
        print('Waveform saving not implemented yet')
        #sd.waveform = 
              
    mua = spy.load(spy_filename, tag = 'spikes')
    sd.info = mua.info
    spy.save(sd, container = spy_filename, tag = 'SUA')

    try:
        change_permissions_recursively(sua_folder)
    except:
        print('Unable to change permissions of files automatically')



