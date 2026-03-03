import os
import numpy as np
from scipy.spatial import KDTree # most efficient for our 3d data
import syncopy as spy
from preprocessing import align_ephys as align
from preprocessing import preprocess_ephys as prep
from preprocessing import snippet_ephys as snip
from parse_logfile import TextLog
from . import rf_mapping as rfm

# make spike and mua and event data
# stim on and stim off is a "trial"
# ? check the timing by aligning the photodiode?
# parse the logfile and get the black and white positions per "trial"
# get average spike rate for full ?200ms? 
# multiple delays,
 
def create_data_folder(spy_filename, root_save = '/cs/departmentN5/neuro/'):

    # make save folder
    if "/OWzeronoise/".lower() in spy_filename.lower() or "/ZeroNoise_MOUSE/".lower() in spy_filename.lower():
        root_save = os.path.join(root_save, 'mouse')
    elif "/MWzeronoise/".lower() in spy_filename.lower() or "/ZeroNoise_MONKEY/".lower() in spy_filename.lower():
        root_save = os.path.join(root_save, 'monkey')
    else:
        print('species unrecognised')
        
    data_folder = root_save
    
    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)
    
        
    return data_folder

def create_figure_folder(spy_filename, data_folder):
    head, tail = os.path.split(spy_filename)
    figure_folder= os.path.join(data_folder, tail[:-4])
    if not os.path.isdir(figure_folder):
        os.makedirs(figure_folder)
    
    return figure_folder



def create_rfmapping_sparse_snippet(compute_folder, alignment):
    aligned_events, basename = align.load_aligned_eventmarkers(compute_folder, alignment)

    # create the container
    container = os.path.join(compute_folder, basename + '_snip' + alignment)

    # correct trials
    tag = "stimon-stimoff"
    MappingStimulusOn = 201
    MappingStimulusOff = 202
    ed = spy.EventData(aligned_events, dimord=["unadjusted_sample","sample","log_times","eventid"], samplerate=align.evt_samplerate)
    ed.definetrial(pre=0, trigger=MappingStimulusOn, stop=MappingStimulusOff)

    # save
    #ed.save(container, tag=tag, overwrite=True)

    return ed

def parse_sparse_noise(archive_folder, alignment):
    logfiles = align.find_logfiles(archive_folder)
    for filename in logfiles:
        if filename.find(alignment) > -1:
            break

    with TextLog(filename) as log:
        log.make_id_struct()
        for this_id in log.all_ids:
            if this_id['name'].startswith('RFMappingFlash_'):
                break
        param, param_ts = log.parse_parameters(this_id['id'], st=this_id['start'], end=this_id['end'])
        #log_evt, screen_ts, evt_desc, log_ts = log.parse_eventmarkers()

    black_ts = param_ts['ShowSparseBlackSpherical']
    black_locs = np.array(param['ShowSparseBlackSpherical'])

    white_ts = param_ts['ShowSparseWhiteSpherical']
    white_locs = np.array(param['ShowSparseWhiteSpherical'])

    # azimuth, elevation, colour, sparse_ts = log.parse_sparsenoise()

    # inc = np.logical_or(log_evt == 201, log_evt == 202)
    # on_ts = screen_ts[inc]
    # off_ts = screen_ts[inc]

    #GridDeltaAngle,GridRadius,GridMinAzimuth,GridMaxAzimuth,GridMinElevation,GridMaxElevation,GradientMaterialMaskRadius,GradientMaterialMaskDensity

    # take the final parameter due to resize post spawn
    azimuth = [param['GridMinAzimuth'][-1], param['GridMaxAzimuth'][-1]]
    elevation = [param['GridMinElevation'][-1], param['GridMaxElevation'][-1]]
    spacing = param['GridDeltaAngle'][-1]

    return black_ts, black_locs, white_ts, white_locs, azimuth, elevation, spacing


def rf_mapping_sparse(archive_folder, make_plots=True, save_plots=True, do_preprocessing=True, use_SUA=False):
    delays = np.arange(40,140,10)/1000 # seconds

    # does spike preprocessing and create alignments
    if do_preprocessing:
        spy_filename, alignments = rfm.do_preprocessing(archive_folder, spikes=True)
    else:
        nwb_filename = prep.convert2nwb(archive_folder)
        spy_filename = nwb_filename[:-4]+'.spy'
        compute_folder, unused = os.path.split(spy_filename)
        alignments = align.find_all_alignments(compute_folder)

    if len(alignments) > 1:
        raise ValueError('More than one alignment for this file, choose which manually')

    compute_folder = os.path.dirname(spy_filename)
    rfEventData = create_rfmapping_sparse_snippet(compute_folder, alignments[0])

    # find logfile containing alignment and _Cont.log
    #stim_azimuth, stim_elevation, colour, screen_ts = parse_sparse_noise(archive_folder, alignments[0])
    out = parse_sparse_noise(archive_folder, alignments[0])
    black_ts, black_locs, white_ts, white_locs, azimuth, elevation, spacing = out
    
    if use_SUA:
        data = spy.load(spy_filename, tag='SUA')
    else:
        data = spy.load(spy_filename, tag='spikes')
    snip.make_trialdefinition(data, rfEventData)

    # align the trials
    ntrials = data.trialdefinition.shape[0]
    log_trials = np.unique(np.append(black_ts,white_ts))

    if len(log_trials) != ntrials:
        raise ValueError('Wrong number of trials in log %i vs event %i'%(log_trials, ntrials))


    # make grid
    azimuth = np.unique(np.append(black_locs[:,0],white_locs[:,0]))
    elevation = np.unique(np.append(black_locs[:,1],white_locs[:,1]))
    grid_az,grid_el = np.meshgrid(azimuth,elevation)

    # make boolean array across trials
    grid_on = np.zeros((ntrials,azimuth.size, elevation.size),dtype=bool)
    for itrl in range(ntrials):
        inc = white_ts == log_trials[itrl]
        az = np.argmin(np.abs(azimuth[None,:] - white_locs[inc,0][:,None]),axis=1)
        el = np.argmin(np.abs(elevation[None,:] - white_locs[inc,1][:,None]),axis=1)
        grid_on[itrl,az,el] = True

    grid_off = np.zeros((ntrials,azimuth.size, elevation.size),dtype=bool)
    for itrl in range(ntrials):
        inc = black_ts == log_trials[itrl]
        az = np.argmin(np.abs(azimuth[None,:] - black_locs[inc,0][:,None]),axis=1)
        el = np.argmin(np.abs(elevation[None,:] - black_locs[inc,1][:,None]),axis=1)
        grid_off[itrl,az,el] = True

    # make "on" and "off" heatmaps for each delay and channel
    if use_SUA:
        nchans = len(data.unit)
        dimord = data.dimord.index("unit")
        bins = np.concatenate([[0],data.unit_idx+0.5])
        names = data.unit
    else:
        nchans = len(data.channel)
        dimord = data.dimord.index("channel")
        bins = np.concatenate([[0],data.channel_idx+0.5])
        names = data.channel
    ndelays = len(delays)
    on_activity, on_std_activity, off_activity, off_std_activity = [np.zeros((ndelays, nchans, azimuth.size, elevation.size)) for _ in range(4)]
    
    max_activity, std_activity, max_az, max_el = [np.zeros((ndelays, nchans)) for _ in range(4)]

    # loop through delays
    original_trialdef = data.trialdefinition
    for idelay,d in enumerate(delays):
        trialdef = rfm.shift_trial_definition(original_trialdef, d*data.samplerate)
        data.trialdefinition = trialdef

        # firing rate per chan and trial
        counts = np.histogram2d(data.data[:,dimord], data.trialid, bins = [bins, np.arange(ntrials+1)])[0]

        # mean all trials efficiently across channels
        count_grid = np.broadcast_to(counts[:,:,None,None], (nchans, ntrials, azimuth.size, elevation.size))

        chan_grid_on = np.broadcast_to(grid_on[None,:,:,:], (nchans, ntrials, azimuth.size, elevation.size))
        on_activity[idelay,:,:,:] = np.mean(count_grid, where=chan_grid_on, axis=1)
        on_std_activity[idelay,:,:,:] = np.std(count_grid, where=chan_grid_on, axis=1)

        chan_grid_off = np.broadcast_to(grid_off[None,:,:,:], (nchans, ntrials, azimuth.size, elevation.size))
        off_activity[idelay,:,:,:] = np.mean(count_grid, where=chan_grid_off, axis=1)
        off_std_activity[idelay,:,:,:] = np.std(count_grid, where=chan_grid_off, axis=1)

        this_activity = off_activity[idelay,:,:,:].reshape((nchans,grid_az.size))
        this_std_activity = off_activity[idelay,:,:,:].reshape((nchans,grid_az.size))
        
        imax = this_activity.argmax(axis=1)
        max_activity[idelay,:] = this_activity.max(axis=1)
        std_activity[idelay,:] = this_std_activity[np.arange(nchans),imax]
        max_az[idelay,:] = 90-grid_az.ravel()[imax] # 0 is center
        max_el[idelay,:] = grid_el.ravel()[imax]
        
    data.trialdefinition = original_trialdef
    
    # save the delay, max firing rate and maximal grid activity for each neuron
    print('Saving')
    
    best_delay = max_activity.argmax(axis=0)
    
    compute_folder, name = os.path.split(spy_filename)
    csvname = os.path.join(compute_folder, '%s.csv'%(name[:-4]))
    
    #create_rf_csv(filename,ChannelNumber,ChannelName,Azimuth,Elevation,MaxResponse,Width,Delay_ms,Direction
    rfm.create_rf_csv(csvname, np.arange(nchans), data.channel, 
                  Azimuth = max_az[best_delay, np.arange(nchans)],
                  Elevation = max_el[best_delay, np.arange(nchans)],
                  MaxResponse = max_activity[best_delay, np.arange(nchans)],
                  StdResponse = std_activity[best_delay, np.arange(nchans)],
                  Width = np.full((nchans), np.nan),
                  Delay_ms = delays[best_delay]*1000,
                  Direction=None) # ms
    
    # make plots
    if make_plots:
        import matplotlib.pyplot as plt
        import scipy.stats as stat
        fig1, ax = plt.subplots(8,8, figsize=(20,15))
        ax = ax.ravel()
        iax = 0
        print('On activity')
        for ichan in range(nchans):
            orig = on_activity[best_delay[ichan],ichan,:,:].T.copy()
            zscore = (orig - np.nanmean(orig))/np.nanstd(orig)
            pval = 1-stat.norm.cdf(zscore)
            #orig[pval>0.05/zscore.size] = np.nan
            h = ax[iax].imshow(orig, 
                extent=(azimuth.min(),azimuth.max(),elevation.min(),elevation.max()))
            plt.colorbar(h, ax=ax[iax])
            ax[iax].set_title('%s %ims'%(names[ichan], delays[best_delay[ichan]]*1000))
            iax+=1
        plt.show()
        
        fig2, ax = plt.subplots(8,8, figsize=(20,15))
        ax = ax.ravel()

        best_delay = max_activity.argmax(axis=0)
        iax = 0
        print('Off activity')
        for ichan in range(nchans):
            orig = off_activity[best_delay[ichan],ichan,:,:].T.copy()
            zscore = (orig - np.nanmean(orig))/np.nanstd(orig)
            pval = 1-stat.norm.cdf(zscore)
            #orig[pval>0.05/zscore.size] = np.nan
            h = ax[iax].imshow(orig, 
                extent=(azimuth.min(),azimuth.max(),elevation.min(),elevation.max()))
            plt.colorbar(h, ax=ax[iax])
            ax[iax].set_title('%s %ims'%(names[ichan], delays[best_delay[ichan]]*1000))
            iax+=1
        plt.show()

        if save_plots:
            data_folder = create_data_folder(spy_filename)
            figure_folder = create_figure_folder(spy_filename, data_folder)
            fig1.savefig(os.path.join(figure_folder, 'on_activity.svg'), transparent = True)
            fig2.savefig(os.path.join(figure_folder, 'off_activity.svg'), transparent = True)
        
        return max_activity, std_activity, max_az, max_el
        
