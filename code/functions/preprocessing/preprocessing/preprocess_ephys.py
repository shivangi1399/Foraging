import os
import h5py
import numpy as np
from oephys2nwb import export2nwb
import syncopy as spy
from acme import ParallelMap

from .preprocessing_utilities import create_target_folder, get_session_info, create_standard_container, change_permissions_recursively

# main functions
def convert2nwb(foldername, target_root='/mnt/hpc_slurm/projects', target_fullfile=None, overwrite=False):
    """
    Takes oe session data converts OE binary to NWB which can be read in by syncopy
    """

    # check existence of directories
    if not os.path.isdir(foldername):
        raise ValueError('foldername given does not exist')
    
    #MBvrtraining/Cosmos/20220428/DividingStimuli/001/Cosmos-20220428-DividingStimuli-001

    if target_fullfile is None:
        
        if not os.path.isdir(target_root):
            raise ValueError('target_root given does not exist')
        
        species, project, subject, date, experiment, session = get_session_info(foldername)

        # create filename 
        target_file = create_standard_container(project, subject, date, experiment, session, '.nwb')
        print(target_file)

        # create target folder
        target_folder = create_target_folder(target_root, species, project, subject, date, experiment, session)
        print(target_folder)

        target_fullfile = os.path.join(target_folder, target_file)

    if os.path.isfile(target_fullfile) and not overwrite:
        print(target_fullfile, ' already exists and overwrite is false, doing nothing')
        return target_fullfile
    elif os.path.isfile(target_fullfile) and overwrite:
        print('Removing ', target_fullfile)
        os.remove(target_fullfile)
        
    target_folder, target_file = os.path.split(target_fullfile)
    if not os.path.isdir(target_folder):
        original_umask = os.umask(0)
        os.makedirs(target_folder, mode=2770, exist_ok=True)
        os.umask(original_umask)

    print('Creating ', target_fullfile)

    export2nwb(foldername, target_fullfile)

    change_permissions_recursively(target_fullfile, 0o770)
    
    return target_fullfile

def convert2syncopy(nwb_filename, memuse=3000):
    spy_filename = nwb_filename[:-4]+'.spy'
    out = spy.load_nwb(nwb_filename, memuse=memuse, container=spy_filename)
    out = list(out.values())

    for data in out:
        if data.__class__.__name__ == "EventData":
            data.data.file["trialdefinition"] = data.trialdefinition
            data.save(filename = data.filename, overwrite=True)
        elif data.__class__.__name__ == "SpikeData":
            data.save(spy_filename, tag='spikes')
        elif data.__class__.__name__ == "AnalogData":
            data.data.file["trialdefinition"] = data.trialdefinition
            data.save(filename = data.filename, overwrite=True)
            
            if len(data.channel) == 8 and data.channel[0].startswith('ADC'):
                new_name = data.filename.replace(data.tag,'ADC')
            else:
                new_name = data.filename.replace(data.tag,'raw')
                
            os.rename(data.filename, new_name)
            os.rename(data.filename+'.info', new_name+'.info')
            
        else:
            print('unsupported syncopy datatype', data.__class__.__name__)


    change_permissions_recursively(spy_filename)


    return spy_filename

def load_raw_data(raw_data, spy_filename):
    if raw_data is None:
        raw_data = spy.load(spy_filename, tag='raw')
    elif isinstance(raw_data, str):
        out_dict = spy.nwb_load(raw_data)
        raw_data = find_spy_raw_data(out_dict)
        if raw_data is None:
            print('nwb filename given was: %s', raw_data) 
            raise ValueError('No raw data (AnalogData with CH channel names) found')
    elif raw_data.__class__.__name__ != "AnalogData":
        raise ValueError('Expected AnalogData as input')
        
    return raw_data
        
def find_spy_raw_data(out_dict):
    out = list(out_dict.values())
    raw_data = None
    for data in out:
        if data.__class__.__name__ == "AnalogData" and data.channel[0].startswith('CH'):
            if raw_data is not None:
                raise ValueError('2nd set of raw data found: %s' %(data.filename))
            raw_data = data
    return raw_data


def create_lfp(spy_filename, raw_data=None,
               downsample_freq = 1000, highpass = 2, lowpass = 300, remove_noise=False, noise_stop=[49.8, 50.2], 
               n_jobs_startup=50, timeout=60, partition="8GBXS"):
    """
    Extracting Local Field Potential (lfp) from the raw data.
    spy_filename : str
        Path to syncopy container
    raw_data : `None` or `str` or `~syncopy.AnalogData`
        None: data will be loaded from spy_filename
        nwb_filename: data will be loaded using `~syncopy.nwb_load`
        A non-empty Syncopy :class:`~syncopy.AnalogData` object
    downsample_freq: `~numpy.float`    
        A floating number to perform the downsampling step.
    highpass: `~numpy.float`    
        A floating number to perform the bandpass filtering step.
    lowpass: `~numpy.float`    
        A floating number to perform the bandpass filtering step.
    remove_noise: `~numpy.bool`
        Option to bandstop filter noise
    noise_stop: List   
        A non-empty (sorted) list of values between which to perform the bandstop noise filtering step.
    n_jobs_startup: `~numpy.int`    
        Acme option.
    timeout: `~numpy.int`    
        Acme option.
    partition: `str`    
        Acme option.

    Returns
    -------
    lfp : `~syncopy.AnalogData`
        The filtered dataset with the same channels as the input raw_data
    """
    
    raw_data = load_raw_data(raw_data, spy_filename)
    
    client = spy.esi_cluster_setup(partition=partition, n_jobs=raw_data.channel.shape[0], n_jobs_startup=n_jobs_startup, timeout=timeout, interactive_wait=1)
    chan_per_worker=1
    if client is None:
        spy.cluster_cleanup()
        import acme
        client = acme.local_cluster_setup(interactive=False)
        
    print('Creating LFP')

    # Bandpass filter
    lfp = spy.preprocessing(raw_data, filter_class='but', freq=[lowpass, highpass], filter_type='bp', order=4, parallel=True, chan_per_worker=chan_per_worker)
    # Optionally remove noise
    if remove_noise:
        lfp = spy.preprocessing(lfp, filter_class='but', freq=noise_stop, filter_type='bs', order=4, parallel=True, chan_per_worker=chan_per_worker)

    # Downsample 
    lfp = spy.resampledata(lfp, resamplefs=downsample_freq, method='downsample', parallel=True, chan_per_worker=chan_per_worker)
    lfp.info = raw_data.info
    spy.cluster_cleanup()

    print('Saving LFP')

    lfp.save(container=spy_filename, tag='lfp')

    
    change_permissions_recursively(spy_filename)
    
    



def create_eMUA(spy_filename, raw_data=None, 
                downsample_freq = 1000, band_cutoffs = [600, 9000], lowFreq = 200, 
                n_jobs_startup=50, timeout=60, partition="16GBXS"):
    
    # Based on https://www.sciencedirect.com/science/article/pii/S2211124721006033
    """
    Extracting a continuous Multi Unit Activity (MUA) trace from raw data, also called envelope of MUA (eMUA).
    spy_filename : str
        Path to syncopy container
    raw_data : `None` or `str` or `~syncopy.AnalogData`
        None: data will be loaded from spy_filename
        nwb_filename: data will be loaded using `~syncopy.nwb_load`
        A non-empty Syncopy :class:`~syncopy.AnalogData` object
    downsample_freq: `~numpy.float`    
        A floating number to perform the downsampling step.
    band_cutoffs: List   
        A non-empty (sorted) list of values between which to perform the bandpass filtering step.
    lowFreq: `~numpy.float`    
        A floating number to perform the lowpass filtering step.
    n_jobs_startup: `~numpy.int`    
        Acme option.
    timeout: `~numpy.int`    
        Acme option.
    partition: `str`    
        Acme option.

    Returns
    -------
    eMUA : `~syncopy.AnalogData`
        The filtered dataset with the same channels as the input raw_data
    """

    raw_data = load_raw_data(raw_data, spy_filename)

    # Perform bandpass filtering between the frequencies of interest and rectify the output. 
    # **We assume that there's already rectification implemented in the preprocessing function, via the "rectify" keyword**
    client = spy.esi_cluster_setup(partition=partition, n_jobs=raw_data.channel.shape[0], n_jobs_startup=n_jobs_startup, timeout=timeout, interactive_wait=1)
    chan_per_worker=1
    if client is None:
        spy.cluster_cleanup()
        import acme
        client = acme.local_cluster_setup(interactive=False)
    
    print('Creating eMUA')
    filtered = spy.preprocessing(raw_data, filter_class='but', order = 4, freq=[band_cutoffs[0], band_cutoffs[1]], filter_type='bp', direction = 'twopass', rectify = True, parallel=True, chan_per_worker=chan_per_worker)

    # Perform lowpass filtering of the resulting rectified (bandpassed) signal coming from the previous step.
    eMUA = spy.preprocessing(filtered, filter_class='but', order = 4, freq=lowFreq, filter_type='lp', direction = 'twopass', parallel=True, chan_per_worker=chan_per_worker)

    # Downsample
    eMUA = spy.resampledata(eMUA, resamplefs=downsample_freq, method='downsample', parallel=True, chan_per_worker=chan_per_worker)
    eMUA.info = raw_data.info
    
    spy.cluster_cleanup()
    
    print('Saving eMUA')
    
    eMUA.save(container=spy_filename, tag='eMUA')
    

    change_permissions_recursively(spy_filename)



    


def _computeSpikes(filtered, q, refract_time):

    threshold = q * np.std(filtered.data,0)
    spikes_locs = []
    lens = np.zeros(filtered.data.shape[1], dtype=np.int64)
    for i, chan in enumerate(filtered.channel):
        spk_tmp = np.where(np.logical_or(filtered.data[:, i] > threshold[i], filtered.data[:, i] < -threshold[i]))[0]
        if len(spk_tmp):
            spiks = spk_tmp[np.append(False, np.diff(spk_tmp) > refract_time)] 
        else:
            spiks = []
        lens[i] = len(spiks)
        spikes_locs.append(spiks)

    return spikes_locs, lens

def _compute_channel_spikes(filename,dataname, ichan, q, refract_time):

    with h5py.File(filename, mode="r") as f:
        chan = f[dataname][:,ichan]

        threshold = q * np.std(chan)
        spk_tmp = np.where(np.logical_or(chan > threshold, chan < -threshold))[0]
        if len(spk_tmp):
            spiks = spk_tmp[np.append(False, np.diff(spk_tmp) > refract_time)] 
        else:
            spiks = []

    return spiks, len(spiks)

def spikesFromRaw(spy_filename, raw_data=None, 
                band_cutoffs = [600, 9000], refract_time = 2, q = 3, 
                n_jobs_startup=20, timeout=60, partition="16GBXS"):
    
    create_spikes(spy_filename, raw_data=raw_data, 
                band_cutoffs = band_cutoffs, refract_time = refract_time, q = q, 
                n_jobs_startup=n_jobs_startup, timeout=timeout, partition=partition)

def create_spikes(spy_filename, raw_data=None, 
                band_cutoffs = [600, 9000], refract_time = 2, q = 3, 
                n_jobs_startup=20, timeout=60, partition="16GBXS", do_median_subtraction=False, median_channels=slice(128,192)):
    
    """
    Extracting Multi Unit Activity (MUA) from raw data, in the form of spikes.
    spy_filename : str
        Path to syncopy container
    raw_data : `None` or `str` or `~syncopy.AnalogData`
        None: data will be loaded from spy_filename
        nwb_filename: data will be loaded using `~syncopy.nwb_load`
        A non-empty Syncopy :class:`~syncopy.AnalogData` object
    downsample_freq: `~numpy.float`    
        A floating number to perform the downsampling step.
    band_cutoffs: List   
        A non-empty (sorted) list of values between which to perform the bandpass filtering step.
    refract_time: `~int`    
        The number of timesteps that have to pass so that we avoid double counting the rise and fall of a peak.
    q: `~numpy.float`
        A floating number to select the threshold (number of times above/below standard deviation of the signal).
    n_jobs_startup: `~numpy.int`
        Acme option.
    timeout: `~numpy.int`    
        Acme option.
    partition: `str`    
        Acme option.

    Returns
    -------
    Spikes : `~syncopy.SpikeData`
        A non-empty Syncopy :class:`~syncopy.SpikeData` object
    """
      
    raw_data = load_raw_data(raw_data, spy_filename)
    
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    
    # special subtraction for noisy V4 arrays
    if do_median_subtraction:
        print('Subtracting median')
        raw_data = subtract_median(raw_data, median_channels)

    # Perform bandpass filtering between the frequencies of interest and rectify the output. 
    # **We assume that there's already rectification implemented in the preprocessing function, via the "rectify" keyword**. In this case, no rectification.
    client = spy.esi_cluster_setup(partition=partition, n_jobs=raw_data.channel.shape[0], n_jobs_startup=n_jobs_startup, timeout=timeout, interactive_wait=1)
    chan_per_worker=1
    if client is None:
        spy.cluster_cleanup()
        import acme
        client = acme.local_cluster_setup(interactive=False) #, mem_per_worker=partition.split('GB')[0]+'GB'
    
    print('Creating spikes')
    
    filtered = spy.preprocessing(raw_data, filter_class='but', order = 4, freq=[band_cutoffs[0], band_cutoffs[1]], filter_type='bp', direction = 'twopass', 
                                 parallel=True, chan_per_worker=chan_per_worker, rectify=False)

    # We have to create the threshold on a channel basis and find where it's crossed.
    pmap = ParallelMap(_compute_channel_spikes, filtered.filename, filtered.data.name, np.arange(raw_data.channel.shape[0]), q, refract_time, n_inputs=raw_data.channel.shape[0], write_worker_results=True)
    with pmap as p:
        h5name = p.compute()
    # load results of parallelmap
    lens = np.zeros(len(h5name),dtype=int)
    spikes_locs = []
    for ii, fname in enumerate(h5name):
        with h5py.File(fname, 'r') as f:
            lens[ii] = np.array(f['result_1'])
            spikes_locs.append(np.array(f['result_0'], dtype=int))
        os.remove(fname)
    #spikes_locs, lens = computeSpikes(filtered, q, refract_time)

    # Preallocation of the channels array.
    flat_channels = np.repeat(a = np.arange(0, lens.size, dtype=int), repeats = lens)

    # We sort the spike list and order the channels based on that (so that each spike corresponds to one [or more] channel[s]).
    spikes_locs = np.concatenate(spikes_locs)
    print('Spikes found: ', spikes_locs.shape)
    argSorted = np.argsort(spikes_locs)
    flat_channels = flat_channels[argSorted]
    spikes_locs = spikes_locs[argSorted]

    # As we're not doing single-units at this point, just feed in 0s.
    units = np.zeros_like(flat_channels)
    
    print('Saving spikes')

    # Create the SpikeData object to save, with the same sampling rate as the input file.
    sd = spy.SpikeData(data = np.column_stack([spikes_locs, flat_channels, units]).astype(int), dimord = ['sample', 'channel', 'unit'], samplerate = raw_data.samplerate)
    sd.info = raw_data.info
    spy.save(sd, container = spy_filename, tag = 'spikes')

    spy.cluster_cleanup()
    

    change_permissions_recursively(spy_filename)


        
def subtract_median(data, inc_channels):
    # calculate median
    nsamples = data.data.shape[0]
    median = np.zeros(nsamples)
    # chunk for speed
    starts = np.linspace(0, nsamples, 200, dtype=int)
    for st, end in zip(starts[:-1],starts[1:]):
        median[st:end] = np.median(data.data[st:end, inc_channels], axis=1)
    
    # create new spy object
    import h5py
    from syncopy.datatype.continuous_data import AnalogData
    
    angData = AnalogData(dimord=AnalogData._defaultDimord)
    angShape = [None, None]
    angShape[angData.dimord.index("time")] = data.data.shape[0]
    angShape[angData.dimord.index("channel")] = data.data.shape[1]
    h5ang = h5py.File(angData.filename, mode="w")
    angDset = h5ang.create_dataset("data", dtype=data.data.dtype, shape=angShape)
    
    # if isinstance(inc_channels, slice):
    #     inc_channels = np.arange(inc_channels.start, inc_channels.stop, inc_channels.step)
    
    # subtract and write
    print('Creating new median subtracted data')
    for st, end in zip(starts[:-1],starts[1:]):
        #print('Chunck %i-%i'%(st,end))
        tmp = data.data[st:end,:]
        tmp[:,inc_channels] -= median[st:end,None]
        angDset[st:end,:] = tmp
    angDset.flush()
    
    angData.data = angDset
    angData.channel = data.channel
    angData.samplerate = data.samplerate
    angData.trialdefinition = np.array([[0, nsamples, 0]])
    angData.info = data.info
    angData.data.file["trialdefinition"] = angData.trialdefinition
    return angData
    
        
# from syncopy.shared.computational_routine import ComputationalRoutine, propagate_properties
# from syncopy.shared.kwarg_decorators import process_io
        
# @process_io
# def subtract_cF(dat, median, noCompute=False, chunkShape=None):

#     """
#     Provides straightforward subtraction 
#     dat : (N, K) :class:`numpy.ndarray`
#         Uniformly sampled multi-channel time-series data
#     subtraction : (N) :class:`numpy.ndarray`
#         Uniformly sampled multi-channel time-series data
#     noCompute : bool
#         If `True`, do not perform actual calculation but
#         instead return expected shape and :class:`numpy.dtype` of output
#         array.
#     Returns
#     -------
#     subtracted : (N, K) :class:`~numpy.ndarray`
#         The subtracted signals
#     Notes
#     -----
#     This method is intended to be used as
#     :meth:`~syncopy.shared.computational_routine.ComputationalRoutine.computeFunction`
#     inside a :class:`~syncopy.shared.computational_routine.ComputationalRoutine`.
#     Thus, input parameters are presumed to be forwarded from a parent metafunction.
#     Consequently, this function does **not** perform any error checking and operates
#     under the assumption that all inputs have been externally validated and cross-checked.
#     """

#     # operation does not change the shape
#     outShape = dat.shape
#     if noCompute:
#         return outShape, np.float32

#     return dat - median


# class Rectify(ComputationalRoutine):

#     """
#     Compute class that performs rectification
#     of :class:`~syncopy.AnalogData` objects
#     Sub-class of :class:`~syncopy.shared.computational_routine.ComputationalRoutine`,
#     see :doc:`/developer/compute_kernels` for technical details on Syncopy's compute
#     classes and metafunctions.
#     See also
#     --------
#     syncopy.preprocessing : parent metafunction
#     """

#     computeFunction = staticmethod(rectify_cF)

#     # 1st argument,the data, gets omitted
#     valid_kws = list(signature(rectify_cF).parameters.keys())[1:]

#     def process_metadata(self, data, out):

#         propagate_properties(data, out)
