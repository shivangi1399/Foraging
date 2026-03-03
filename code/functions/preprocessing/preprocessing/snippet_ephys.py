import os
import numpy as np
import syncopy as spy

from .align_ephys import load_aligned_eventmarkers, evt_samplerate

def create_log_trialdefinition(aligned_evts):
    # create 1 trial which is aligned to log start
    screen_col = 1
    sample_col = 0

    # find first not nan screen time
    for ii in range(aligned_evts.shape[0]):
        screen = aligned_evts[ii,screen_col]
        if not np.isnan(screen):
            start = screen
            break

    # find last not nan screen time
    for ii in range(-1,-aligned_evts.shape[0],-1):
        screen = aligned_evts[ii,screen_col]
        if not np.isnan(screen):
            end = screen
            break

    return np.array((start, end, 0))

def create_common_snippets(compute_folder, alignment, overwrite=False):

    aligned_events, basename = load_aligned_eventmarkers(compute_folder, alignment)

    evts = aligned_events[:,-1]

    # create the container
    container = os.path.join(compute_folder, basename + '_snip' + alignment)

    # start-end
    tag = "start0-end0"

    ed = create_trialdefinition(aligned_events, pre=None, post=None, start=[3000], trigger=None, stop=[3090], 
                           return_EventData=True)
    ed.save(container, tag=tag, overwrite=overwrite)

    # start-rew
    tag = "start0-rew0"
    trldef = create_uneven_trldef(aligned_events, start=[3000], trigger=None, stop=[5000,6000], stop_screentime=False)
    ed = spy.EventData(aligned_events[:,[1,3]],trialdefinition=trldef, samplerate=evt_samplerate)
    ed.save(container, tag=tag, overwrite=overwrite)

def create_uneven_trldef(aligned_events, start=[3000], trigger=None, stop=[3090],start_screentime=True, stop_screentime=True, trigger_screentime=True):
    # if there is not necessarily a start, end or trigger on every trial then we need to do this via search sorted
    if start is None and trigger is None:
        raise ValueError('Must specify start or trigger')
    elif stop is None and trigger is None:
        raise ValueError('Must specify stop or trigger')

    # get screen or orig times
    evts = aligned_events[:,3]
    def return_times(is_screen):
        if is_screen:
            return aligned_events[:,1]
        else:
            return aligned_events[:,0]
    t_start = return_times(start_screentime)
    t_trigger = return_times(trigger_screentime)
    t_stop = return_times(stop_screentime)

    # get matching times
    def included_times(ievt, times):
        if ievt is None:
            return None
        if len(ievt) == 2:
            return times[np.logical_and(evts>=ievt[0], evts<ievt[1])]
        else:
            return times[evts == ievt[0]]

    t_start = included_times(start, t_start)
    t_trigger = included_times(trigger, t_trigger)
    t_stop = included_times(stop, t_stop)
    
    # remove times that don't align
    def align_times(before,after):
        if len(before) <= len(after):
            inc_afters = np.searchsorted(before, after)
            #remove repeats
            inc_afters = np.unique(inc_afters)
            if inc_afters[0] == 0:
                inc_afters[1:]
            after = after[inc_afters]

        inc_befores = np.searchsorted(after,before)
        # keep the final one of duplicate before 
        before = before[np.diff(inc_befores, append=after.size) > 0]

        if len(before) != len(after):
            before, after = align_times(before,after)

        return before, after
    
    if t_start is not None and t_stop is not None:
        t_start, t_stop = align_times(t_start, t_stop)
        if t_trigger is None:
            t_trigger = t_start.copy()
        else:
            t_trigger, t_stop = align_times(t_trigger, t_stop)
            if len(t_start) > len(t_stop):
                # redo alignment if we lost trials due to missing trigger
                t_start, t_stop = align_times(t_start, t_stop)
    elif t_start is None:
        t_trigger, t_stop = align_times(t_trigger, t_stop)
        t_start = t_trigger.copy()
    elif t_stop is None:
        t_start, t_trigger = align_times(t_start, t_trigger)
        t_stop = t_trigger.copy()

    # align trigger relative to start
    t_trigger = t_start - t_trigger

    return np.column_stack((t_start, t_stop, t_trigger)).astype(int)

    

def create_trialdefinition(aligned_evts, pre=None, post=None, start=[3000], trigger=None, stop=[3090], 
                           start_screentime=True, stop_screentime=True, trigger_screentime=True, 
                           return_EventData=False):
    # create a trialdefinition from aligned_events using either screentime or cpu time
    origsample = aligned_evts[:,0]
    screensample = aligned_evts[:,1]
    tlog = aligned_evts[:,2]
    evts = aligned_evts[:,3]

    n_evts = aligned_evts.shape[0]

    evt_data = np.column_stack((screensample, evts)) # default

    # get matching events
    def included_evts(ievt):
        if ievt is None:
            return None
        if len(ievt) == 2:
            return np.logical_and(evts>=ievt[0], evts<ievt[1])
        else:
            return evts == ievt[0]

    # replace times if necessary
    if not start_screentime:
        inc_start = included_evts(start)
        evt_data[inc_start,0] = origsample[inc_start]
    if not stop_screentime:
        inc_stop = included_evts(stop)
        evt_data[inc_stop,0] = origsample[inc_stop]
    if not trigger_screentime:
        inc_trigger = included_evts(trigger)
        evt_data[inc_trigger,0] = origsample[inc_trigger]

    evt_spy = spy.EventData(evt_data, samplerate=evt_samplerate)
    evt_spy.definetrial(pre=pre, post=post, start=start, trigger=trigger, stop=stop)
    if return_EventData:
        return evt_spy
    else:
        return evt_spy.trialdefinition


def make_trialdefinition(data, eventData):
    if not hasattr(data, 'info'):
        print('Need to upgrade syncopy for starting_time information')
        raise AttributeError(name='info', obj=data)
        
    trldef = eventData.trialdefinition
    try:
        trldef[:,:2] -= int(data.info['starting_time']*eventData.samplerate)
    except KeyError:
        err_msg = """data is missing starting_time, try to load from the .nwb file as follows:
        import pynwb
        nwb_filename = spy_filename[:-4]+'.nwb'
        nwbfile = pynwb.NWBHDF5IO(nwb_filename, "r", load_namespaces=True).read()
        data.info['starting_time']=nwbfile.acquisition["ElectricalSeries_1"].starting_time """
        print(err_msg)
        raise KeyError('data is missing starting_time, please load from the .nwb file')
    if eventData.samplerate == data.samplerate:
        data.trialdefinition = trldef
    else:
        data.trialdefinition = ((trldef/eventData.samplerate)*data.samplerate).astype(int)
    

def snippet_data(spy_container, data_tag='eMUA', alignment='150714', event_tag=None, save_data=False):
    # alignment should be a string uniquely specifying the eventmarker alignment
    # event_tag should be given when a premade EventData with trialdefinition should be loaded

    compute_folder = os.path.dirname(spy_container)

    for (dirpath, dirnames, filenames) in os.walk(compute_folder):
        break

    snip_containers = []

    # load the trialdefinition
    if event_tag is None: # create 1 big trial for this alignment
        # find the eventmarkers
        aligned_events = None
        for eventfile in filenames:
            if eventfile.split('.')[-2].endswith(alignment):
                if aligned_events is None:
                    aligned_events = np.load(eventfile)
                else:
                    raise ValueError('Specified alignment %s matched multiple files'%(alignment))

        if aligned_events is None:
            raise ValueError('No aligned eventmarkers %s found, create first with snippet_eventmarkers'%(alignment))
        # create the trialdefinition
        trialdef = create_log_trialdefinition(aligned_events)
        events = spy.EventData(aligned_events[:,[1,3]],trialdefinition=trialdef, samplerate=evt_samplerate)
    
    else: # load EventData with premade trialdefinition

        for directory in dirnames:
            if directory.endswith(alignment+'.spy'):
                snip_containers.append(directory)

        if len(snip_containers) is None:
            raise ValueError('No snip containers found, create first with snippet_eventmarkers')

        if len(snip_containers) > 1:
            raise ValueError('Too many snip containers match given alignment %s'%(alignment))

        # load snip_events
        events = spy.load(os.path.join(compute_folder, snip_containers[0]), tag=event_tag)

    # load data
    data = spy.load(spy_container, tag=data_tag)
    make_trialdefinition(data, events)

    if save_data:
        if event_tag is None:
            snip_container = spy_container[:-4] + '_snip' + alignment + spy_container[-4:]
            data.save(os.path.join(compute_folder, snip_container[0]), tag=data_tag)
        else:
            data.save(os.path.join(compute_folder, snip_container[0]), tag=data_tag+'-'+event_tag)
    else:
        return data

# def snippet_syncopy_data(analog, events):
    
#     shorter = analog.copy(deep=True)

#     data_group = getattr(shorter, "_data")
#     #del h5py.File(shorter.filename, 'R+')['data']
#     del data_group
#     shorter._set_dataset_property_with_ndarray(analog.data[], 'data', 2)

    

