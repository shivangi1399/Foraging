import os
import numpy as np
from scipy.spatial import KDTree # most efficient for our 3d data
from h5py import File
import syncopy as spy
from acme import ParallelMap, esi_cluster_setup
from preprocessing import preprocess_ephys as prep
from preprocessing import align_ephys as align
from preprocessing import snippet_ephys as snip
from . import dome_backproject as dbp

def computePSTH(trialtime, trialid, units, bins, ntrials):
    """
    Computing Peri-Stimulus Time Histograms (PSTHs) from spike data. Every time quantity is expected to be 
    in seconds.
    trialtime : `~numpy.array`
        A non-empty NumPy array, when a spike took place relative to trial start in seconds.
    trialid : `~numpy.array`
        A non-empty NumPy array, trial id of a spike.
    units: `~numpy.array`
        A non-empty NumPy array, to which unit each spike belonged.
    bins: `~numpy.array`
        A non-empty NumPy array of monotonically increasing array of PSTH bin edges including the rightmost edge.
    ntrials: `int`
        Total number of possible trials in the trialid
    Returns
    -------
    counts : `~numpy.array`
        A (nunits, nbins, ntrials) matrix (tensor) that denotes the spike counts for each trial.
    """
    
    nbins = bins.size-1
    uniq_units = np.unique(units)
    counts = np.zeros((len(uniq_units),nbins, ntrials))
    for ii, iunit in enumerate(uniq_units):
        idx_units = units == iunit
        if np.any(idx_units):
            counts[ii,:,:] = np.histogram2d(trialtime[idx_units], trialid[idx_units], bins = [bins, np.arange(ntrials+1)])[0]
        else:
            counts[ii,:,:] = np.zeros(nbins,ntrials)
    
    return counts

def computePSTHchan(trialtime, chanid, units, bins, nchans, nunits, remove_artifacts=False):
    """
    Computing Peri-Stimulus Time Histograms (PSTHs) from spike data. Every time quantity is expected to be 
    in seconds.
    trialtime : `~numpy.array`
        A non-empty NumPy array, when a spike took place relative to trial start in seconds.
    trialid : `~numpy.array`
        A non-empty NumPy array, trial id of a spike.
    units: `~numpy.array`
        A non-empty NumPy array, to which unit each spike belonged.
    bins: `~numpy.array`
        A non-empty NumPy array of monotonically increasing array of PSTH bin edges including the rightmost edge.
    ntrials: `int`
        Total number of possible trials in the trialid
    Returns
    -------
    counts : `~numpy.array`
        A (nunits, nbins, ntrials) matrix (tensor) that denotes the spike counts for each trial.
    """
    
    if remove_artifacts:
        ndiff = 32
        ts = 0.002 #sec
        
        too_many = (trialtime[ndiff:] - trialtime[:-ndiff]) < ts
        gap = np.diff(too_many.astype(int))
        
        artifact_sts = np.nonzero(gap == 1)[0]
        if too_many[0]:
            artifact_sts = np.append(0, artifact_sts)
        artifact_ends = np.nonzero(gap == -1)[0] + ndiff
        if len(artifact_ends) < len(artifact_sts):
            artifact_ends = np.append(artifact_ends, len(too_many)-1)
            
        chanid = chanid.astype(float)
        for iart in range(len(artifact_sts)):
            trialtime[artifact_sts[iart]:artifact_ends[iart]] = np.nan
            chanid[artifact_sts[iart]:artifact_ends[iart]] = np.nan
    
    nbins = bins.size-1
    
    if nunits == 1:
        counts = np.histogram2d(trialtime, chanid, bins = [bins, np.arange(nchans+1)])[0]
        return counts[None,:,:]
    
    uniq_units = np.unique(units)
    counts = np.zeros((len(uniq_units),nbins, nchans))
    
    for ii, iunit in enumerate(uniq_units):
        idx_units = units == iunit
        if np.any(idx_units):
            counts[ii,:,:] = np.histogram2d(trialtime[idx_units], chanid[idx_units], bins = [bins, np.arange(nchans+1)])[0]
        else:
            counts[ii,:,:] = np.zeros(nbins,nchans)
    
    return counts

def psth_per_trial(data, chan_dim, unit_dim, trialtime, trialid, this_trl, bins, nchans, nunits):
    nbins = bins.size-1
    
    inc = trialid == this_trl
    if np.all(inc == 0):
        return np.zeros((nbins, nchans))
    # assumes sorted data so trials are continuous
    st = inc.argmax()
    end = inc[st:].argmin()+st
    s_trl = np.s_[st:end]
    
    return computePSTHchan(trialtime[s_trl], 
                          data[s_trl,chan_dim], data[s_trl,unit_dim],
                          bins, nchans, nunits)[0,:,:]

def psth_per_channel_and_direction(spike, bins):
    # returns list per direction of psths for each channel
    nbins = bins.size-1
    ntrials = spike.trialinfo.shape[0]
    chan_dim = [x for x,y in enumerate(spike.dimord) if y == 'channel'][0]
    unit_dim = [x for x,y in enumerate(spike.dimord) if y == 'unit'][0]
    nchans = len(spike.channel)
    if spike.data[:,unit_dim].max() > 0:
        nunits = len(np.unique(spike.data[:,unit_dim]))
    else:
        nunits = 1
    trialtime = spike.trialtime
    trialid = spike.trialid
    
    psth_all = np.zeros((nbins, nchans, ntrials))
    
    for this_trl in range(ntrials):
        tmp = psth_per_trial(spike.data, chan_dim, unit_dim, trialtime, trialid, this_trl, bins, nchans, nunits)
        psth_all[:,:,this_trl] = tmp
    
    trialdir = spike.trialinfo[:,0]
    all_dirs = np.unique(trialdir)
    
    print('Averaging directions')
    
    psth = [] # nchans, nbins
    
    #dir_trls = []
    for idir, direction in enumerate(all_dirs):
        #dir_trls.append(np.nonzero(trialdir == direction)[0])
        psth.append(np.nanmean(psth_all[:,:,trialdir == direction], axis=2).T)
            
    return psth

def compute_avg_muau(data, downsample=60,nframes=120):
    np.round((np.linspace(0,1.8,nframes)/data.samplerate) * downsample).astype(int)

def shift_trial_definition(trldef, samp_shift):
    out = trldef.copy()
    out[:,0:2] = out[:,0:2]+samp_shift
    return out

def get_eventinfo_per_trial(ed, evtst, evtend):
    evt_idx = ed.dimord.index("eventid")
    return [t[np.logical_and(t[:,evt_idx]>=evtst, t[:,evt_idx]<=evtend),evt_idx].astype(np.float64) - evtst for t in ed.trials]

def create_rfmapping_bar_snippet(compute_folder, alignment):
    aligned_events, basename = align.load_aligned_eventmarkers(compute_folder, alignment)

    # create the container
    container = os.path.join(compute_folder, basename + '_snip' + alignment)

    # correct trials
    tag = "start-stimon-rew"
    MappingStimulusOn = 201
    trldef = snip.create_uneven_trldef(aligned_events, start=[3000], trigger=[MappingStimulusOn], stop=[5000,6000])
    ed = spy.EventData(aligned_events, dimord=["unadjusted_sample","sample","log_times","eventid"], trialdefinition=trldef, samplerate=snip.evt_samplerate)

    # find the direction within each trial
    direction = get_eventinfo_per_trial(ed, 12000, 12500)
    for itrl, d in enumerate(direction):
        if len(d) != 1:
            raise ValueError('%i directions on trial %i'%(len(d),itrl))
    direction = np.array(direction)

    # add the extra information into the trldef
    trldef = np.concatenate((trldef, direction),axis=1)
    ed.trialdefinition = trldef

    # save
    ed.save(container, tag=tag, overwrite=True)

    return ed

def run_preprocessing(archive_folder, lfp=False, eMUA=False, spikes=False):
    nwb_filename = prep.convert2nwb(archive_folder)
    spy_filename = prep.convert2syncopy(nwb_filename)

    alignments = align.snippet_eventmarkers(archive_folder)

    if lfp:
        prep.create_lfp(spy_filename)
    if eMUA:
        prep.create_eMUA(spy_filename)
    if spikes:
        prep.spikesFromRaw(spy_filename)
    

    return spy_filename, alignments

def create_bin_edges(ed):
    # use the average time of each frame flip to determine the size of the bins
    evt_idx = ed.dimord.index("eventid")
    evtst = 12600
    evtend = 12800
    flip_times = []
    for times, data in zip(ed.time, ed.trials):
        inc_evt = np.logical_and(data[:,evt_idx]>=evtst, data[:,evt_idx]<=evtend)
        angle_times = times[inc_evt]
        
        # since the first time seems to be wrong replace with 996s
        flips = times[data[:,evt_idx] == 996]
        t0 = np.argmin(np.abs(flips - angle_times[0]))
        inc_times = flips[t0:t0+len(angle_times)+1] # plus 1 for bin edges
        if np.diff(inc_times).max() < 0.018: # exclude trials with frame skips
            flip_times.append(inc_times[:,None])
            
    szs = np.array([x.shape[0] for x in flip_times])
    nflips = np.median(szs)
    if not np.all(szs == nflips):
        print('Not all trials have %i frames'%(nflips))
        print(szs[szs != nflips])
        flip_times = [x for x in flip_times if x.shape[0]==nflips]

    flip_times = np.concatenate(flip_times,axis=1)
        
    bins = np.mean(flip_times, axis=1)

    return bins

# def get_angle_per_frame(ed):
#     # find the radial coordinate within each trial
#     angles = get_eventinfo_per_trial(ed, 12600, 12800)

#     trialdir = ed.trialinfo[:,0]
#     all_dirs = np.unique(trialdir)
#     ndirs = len(all_dirs)

#     dir_angles = []

#     for idir, direction in enumerate(all_dirs):
#         dirtrl = np.nonzero(trialdir == direction)[0]
#         dir_angles.append(np.mean(np.concatenate([angles[itrl] for itrl in dirtrl], axis=1),axis=1))

#     return dir_angles

def all_rows_equal(arr):
    return np.all(np.all(arr == arr[0:1,:], axis=1))

def get_angle_per_direction(speed, direction, radial_pos, bins):

    # find angles for each trial
    start_pos = np.array([x[0] for x in radial_pos])
    sign = np.array([x[5] - x[0] for x in radial_pos], dtype=float)
    sign /= abs(sign)

    if not np.all(sign == sign[0]):
        raise ValueError('Not all signs are the same, parameters were changed part way through recording, not handled')

    deg = speed*bins[None,:-1] # exclude final bin edge
    
    angles = start_pos[:,None]+(deg*sign[0])

    # reduce to 1 output per direction
    dirs, idirs = np.unique(direction, return_index=True)

    # check they all match
    for this_dir in dirs:
        inc = this_dir == direction.ravel()
        if not all_rows_equal(angles[inc,:]):
            raise ValueError('Not all angles are the same, parameters were changed part way through recording, not handled')

    return angles[idirs,:]
    

def get_bar_info(ed):
    # get corrected direction for rounding in eventmarkers
    direction = np.array(get_eventinfo_per_trial(ed, 12000, 12500))
    uniq_dirs, dir_idx = np.unique(direction, return_inverse=True)
    corrected_dirs = np.arange(0,360,360/len(uniq_dirs))
    if np.any(abs(uniq_dirs-corrected_dirs) >= 1):
        raise ValueError('Directions are not calculated as expected, update code')
    direction = corrected_dirs[dir_idx]

    # get other info
    radial_pos = [x - 180 for x in get_eventinfo_per_trial(ed, 12500, 12860)]
    colour = np.array(get_eventinfo_per_trial(ed, 13000, 13100))
    thickness = np.array(get_eventinfo_per_trial(ed, 13200, 13400))/10
    speed = np.array(get_eventinfo_per_trial(ed, 13400, 14400))/10 #deg per second

    return direction, radial_pos, colour, thickness, speed

def calc_grid_overlap(uniq_dirs, bar_width, grid_az, grid_el, angles):
    # loop through each direction calculating the grid overlap points of sweep
    
    #ndirs = len(uniq_dirs)

    overlap = []
    for idir, this_dir in enumerate(uniq_dirs):
        # get top, middle and bottom points for each frame of sweep
        nsteps = 3
        # for small angle (22.5) sweeping the right side of dome starts from -5 (near fixation) -> 50 (far right)
        # in standard spherical (center is (90,90) deg) this means we multiply by -1 to reverse direction
        points = dbp.get_sweep_direction_points(this_dir, angles[idir,:]*-1, steps=nsteps)

        top_lon, top_lat = dbp.cartesian2geo(points[0:1,:]) # all top points identical, fixed rotation
        mid_lon, mid_lat = dbp.cartesian2geo(points[1::nsteps,:])

        # convert azimuth to longitude
        grid_lon = 90 - grid_az.copy()
        grid_lat = grid_el.copy()

        dist = dbp.dist_grid2sweeps(np.array([top_lon[0], top_lat[0]]),
                                np.column_stack((mid_lon, mid_lat)),
                                np.column_stack((grid_lon, grid_lat)))

        overlap.append(dist < bar_width)

    return overlap


def calc_mean_activity(spy_filename, ed_filename, bins, overlap, d=0):
    
    data = spy.load(spy_filename, tag='spikes')
    rfEventData = spy.load(ed_filename)
    snip.make_trialdefinition(data,rfEventData)

    # need to test if I should calculate psth in here or better using syncopy
    trialdef = shift_trial_definition(data.trialdefinition, d*data.samplerate)
    data.trialdefinition = trialdef
    
    psth = psth_per_channel_and_direction(data, bins)

    ndirs = len(psth)
    nchans = len(data.channel)#psth[0].shape[0]

    nsweeps, ngrid = overlap[0].shape
    
    print('Calculating activity per grid points')

    # loop through each direction calculating activity per grid point
    grid_dir_activity = np.full((ndirs,nchans,ngrid), np.nan)
    for idir in range(ndirs):
        dir_psth = psth[idir]

        # need to mean all sweeps efficiently across channels
        point_activity = np.broadcast_to(dir_psth[:,:,None], (dir_psth.shape[0],dir_psth.shape[1],ngrid))
        chan_inc_sweeps = np.broadcast_to(overlap[idir][None,:,:], (nchans,nsweeps,ngrid))

        grid_dir_activity[idir,:,:] = np.mean(point_activity,where=chan_inc_sweeps,axis=1)

    # Replace non overlapped points with spontaneous activity 
    non_overlapped = np.isnan(grid_dir_activity)
    spontaneous = np.mean(np.array([np.median(x, axis=1) for x in psth]), axis=0)
    spontaneous = np.broadcast_to(spontaneous[None,:,None], (ndirs, nchans, ngrid))
    grid_dir_activity[non_overlapped] = spontaneous[non_overlapped]

    # Calc activity
    grid_activity = np.mean(grid_dir_activity,axis=0) # nchans x ngrid
    imax = grid_activity.argmax(axis=1)
    std_activity = np.std(grid_dir_activity[:,np.arange(nchans),imax],axis=0)

    return grid_activity, std_activity

def create_rf_csv(filename,ChannelNumber,ChannelName,Azimuth,Elevation,MaxResponse,StdResponse,Width,Delay_ms,Direction=None):
    nchans = ChannelNumber.size
    header = 'ChannelNumber,ChannelName,Azimuth,Elevation,MaxResponse,StdResponse,Width,Delay_ms,Direction'


    #dtype = [('Name', (np.str_, 10)), ('Marks', np.float64), ('GradeLevel', np.int32)]
    dtype = [   ('ChannelNumber', np.int32), 
                ('ChannelName',(np.str_, 20)),
                ('Azimuth',np.float64),
                ('Elevation',np.float64),
                ('MaxResponse',np.float64),
                ('StdResponse',np.float64),
                ('Width',np.float64),
                ('Delay_ms',np.float64), # ms
                ('Direction',np.float64)]

    fmt = [ '%i',
            '%s',
            '%.3f',
            '%.3f',
            '%.3f',
            '%.3f',
            '%.3f',
            '%.1f', #ms
            '%.3f']

    #Creating a Strucured Numpy array
    #structuredArr = np.array([('Sam', 33.3, 3), ('Mike', 44.4, 5), ('Aadi', 66.6, 6), ('Riti', 88.8, 7)], dtype=dtype)
    rfs = np.empty((nchans), dtype=dtype)
    rfs['ChannelNumber'] = ChannelNumber
    rfs['ChannelName'] = ChannelName
    rfs['Azimuth'] = Azimuth
    rfs['Elevation'] = Elevation
    rfs['MaxResponse'] = MaxResponse
    rfs['StdResponse'] = StdResponse
    rfs['Width'] = Width
    rfs['Delay_ms'] = Delay_ms

    if Direction is not None:
        rfs['Direction'] = Direction
    else:
        rfs['Direction'] = np.nan

    np.savetxt(filename, rfs, fmt=fmt, delimiter=',', header=header, comments='')
    
def plot_rfs(filename):
    import matplotlib.pyplot as plt
    
    rfs = np.load(filename)
    figname = os.path.join(filename[:-4] + '_activity')
    
    print('Plotting')

    max_subplots = 32
    
    nchans = rfs['max_activity'].shape[0]
    nfigs = (nchans-1)//max_subplots
    extent = [rfs['grid_azimuth'].min(), rfs['grid_azimuth'].max(),
              rfs['grid_elevation'].min(), rfs['grid_elevation'].max()]

    for ifig in range(nfigs+1):
        # figure properties
        if ifig == nfigs:
            fig_chans = (nchans - ifig*max_subplots)%max_subplots
            if fig_chans == 0:
                fig_chans = max_subplots
        else:
            fig_chans = max_subplots

        cols = np.ceil(np.sqrt(fig_chans)).astype(int)
        rows = np.ceil(fig_chans/cols).astype(int)
        
        # do plotting
        fig, axs = plt.subplots(rows,cols, figsize=[10,10])
        axs = axs.ravel()

        for iax, ichan in enumerate(range(ifig*max_subplots, ifig*max_subplots+fig_chans)):
            axs[iax].imshow(rfs['max_activity'][ichan,:,:], extent=extent, origin='lower')
        fig.savefig('%s_%i'%(figname,ifig)+'.png')
        fig.savefig('%s_%i'%(figname,ifig)+'.pdf')



def rf_mapping_bar(archive_folder, spy_filename=None, make_plots=True, do_preprocessing=True):
    delays = np.arange(40,140, 10)/1000 # seconds

    # does spike preprocessing and create alignments
    if do_preprocessing:
        spy_filename, alignments = run_preprocessing(archive_folder, spikes=True)
    else:
        if spy_filename == None:
            nwb_filename = prep.convert2nwb(archive_folder)
            spy_filename = nwb_filename[:-4]+'.spy'
        compute_folder = os.path.dirname(spy_filename)
        alignments = align.find_all_alignments(compute_folder)

    if len(alignments) > 1:
        raise ValueError('More than one alignment for this file, choose which manually')

    compute_folder = os.path.dirname(spy_filename)
    rfEventData = create_rfmapping_bar_snippet(compute_folder, alignments[0])

    data = spy.load(spy_filename, tag='spikes')
    
    snip.make_trialdefinition(data,rfEventData)
    
    
    bins = create_bin_edges(rfEventData)
    
    direction, radial_position, colour, thickness, speed = get_bar_info(rfEventData)
    angles = get_angle_per_direction(speed, direction, radial_position, bins)
    
    if angles.shape[1] < len(bins)-1:
        raise ValueError('angles %i less than bins %i'%(angles.shape[1], len(bins)-1))
    elif angles.shape[1] > len(bins)-1:
        raise ValueError('angles %i greater than bins %i'%(angles.shape[1], len(bins)-1))
    
    print('Find dome overlaps')

    # create grid to evaluate
    grid_deg = 0.1
    azimuth = np.arange(-15,15, grid_deg)
    elevation = np.arange(-10,10, grid_deg)
    grid_az,grid_el = np.meshgrid(azimuth,elevation)

    # calculate grid overlap per direction
    uniq_dirs = np.unique(direction)
    bar_width = dbp.distance(0,0,0,np.deg2rad(np.mean(thickness)/2))
    overlap = calc_grid_overlap(uniq_dirs, bar_width, grid_az.ravel(), grid_el.ravel(), angles)

    print('Looping through delays')
    
    # calculate grid activity per channel for multiple different delays
    nchans = len(data.channel)
    ngrid = grid_az.size
        
    all_grid_activity = np.zeros((len(delays), nchans, ngrid))
    std_activity = np.zeros((len(delays), nchans))
    max_activity = np.zeros((len(delays), nchans))
    max_az = np.zeros((len(delays), nchans))
    max_el = np.zeros((len(delays), nchans))
        
    #try to setup client
    client = esi_cluster_setup(n_jobs_startup=4, n_jobs=len(delays), interactive_wait=1)
    
    if client is not None:
        with ParallelMap(calc_mean_activity, spy_filename, rfEventData.filename, bins, overlap, d=delays, n_inputs=len(delays)) as pmap:
            filenames = pmap.compute()
        client.shutdown()

        for idelay, fname in enumerate(filenames):
            with File(fname, 'r') as f:
                all_grid_activity[idelay,:] = np.array(f['result_0'])
                std_activity[idelay,:] = np.array(f['result_1'])

                grid_activity = all_grid_activity[idelay,:]

                imax = grid_activity.argmax(axis=1)
                max_activity[idelay,:] = grid_activity.max(axis=1)
                max_az[idelay,:] = grid_az.ravel()[imax] # 0 is center
                max_el[idelay,:] = grid_el.ravel()[imax]
                
    else:
        for idelay, d in enumerate(delays):
            print('Delay ', d)
            grid_activity, std_act = calc_mean_activity(spy_filename, rfEventData.filename, bins, overlap, delays)
            all_grid_activity[idelay,:] = grid_activity
            std_activity[idelay,:] = std_act

            imax = grid_activity.argmax(axis=1)
            max_activity[idelay,:] = grid_activity.max(axis=1)
            max_az[idelay,:] = grid_az.ravel()[imax] # 0 is center
            max_el[idelay,:] = grid_el.ravel()[imax]
            
    # calculate rf properties
    best_delay = max_activity.argmax(axis=0) # earliest max delay
    
    # do smoothing, fit elipse and calculate center, width and direction
    #

    # save the delay, max firing rate and maximal grid activity for each neuron
    print('Saving')
    
    compute_folder, name = os.path.split(spy_filename)
    
    csvname = os.path.join(compute_folder, '%s.csv'%(name[:-4]))
    maxname = os.path.join(compute_folder, '%s'%(name[:-4]))

    #create_rf_csv(filename,ChannelNumber,ChannelName,Azimuth,Elevation,MaxResponse,Width,Delay_ms,Direction
    create_rf_csv(csvname, np.arange(nchans), data.channel, 
                  Azimuth = max_az[best_delay, np.arange(nchans)],
                  Elevation = max_el[best_delay, np.arange(nchans)],
                  MaxResponse = max_activity[best_delay, np.arange(nchans)],
                  StdResponse = std_activity[best_delay, np.arange(nchans)],
                  Width = np.full((nchans), np.nan),
                  Delay_ms = delays[best_delay]*1000,
                  Direction=None) # ms

    np.savez(maxname, 
            max_activity=all_grid_activity[best_delay,np.arange(nchans),:].reshape(nchans, grid_az.shape[0], grid_el.shape[1]), 
            grid_azimuth=azimuth, # 0 is center
            grid_elevation=elevation)

    # make plots
    if make_plots:
        print('Plotting')
        plot_rfs(maxname + '.npz')
        
