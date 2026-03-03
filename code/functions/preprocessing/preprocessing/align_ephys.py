import os
import glob
from datetime import datetime
import numpy as np
from open_ephys.analysis import Session
from parse_logfile import TextLog

from preprocessing.preprocessing_utilities import get_session_info, create_target_folder, change_permissions_recursively


evt_samplerate = 30000

def snippet_eventmarkers(archive_folder, target_root='/mnt/hpc_slurm/projects'):

    logfiles = find_logfiles(archive_folder)
    oe_recording = find_oe_recording(archive_folder)

    # create save folder
    species, project, subject, date, experiment, session = get_session_info(archive_folder)
    save_folder = create_target_folder(target_root, species, project, subject, date, experiment, session)
    if not os.path.isdir(save_folder):
        original_umask = os.umask(0)
        os.makedirs(save_folder, mode=2770, exist_ok=True)
        os.umask(original_umask)
    
    # load oe eventmarkers
    oe_session = Session(oe_recording)
    if len(oe_session.recordnodes[0].recordings) > 1:
        raise ValueError("Multiple openephys recordings detected, code cant handle this")

    this_recording = oe_session.recordnodes[0].recordings[0]
    evt, ts = create_oe_eventmarkers(this_recording)
    samplerate = this_recording.info['events'][0]['sample_rate']

    if samplerate != evt_samplerate:
        raise ValueError('This samplerate %0.1f does not match default samplerate %0.1f, code cannot handle this'%(samplerate, evt_samplerate))
        
    alignments = []

    if len(logfiles) == 1:
        # this is the simple usual case
        with TextLog(logfiles[0]) as log:
            log_evt, screen_ts, evt_desc, log_ts = log.parse_eventmarkers()
            align = format_iso8601(log.start_time)
        
        aligned_evts = align_eventmarkers_to_screen(evt, ts, log_evt, log_ts, screen_ts)

        # save events
        eventsfile = "-".join([project, subject, date, experiment, session]) + "_events" + align
        np.save(os.path.join(save_folder, eventsfile), aligned_evts)
        alignments.append(align)
    else:
        # we need to figure out which logfile is which
        for ilog in logfiles:
            print(ilog)
            with TextLog(ilog) as log:
                log_evt, screen_ts, evt_desc, log_ts = log.parse_eventmarkers()
                align = format_iso8601(log.start_time)

            evt_match, ts_match = align_eventmarkers_to_log(evt, ts, log_evt, log_ts, screen_ts)

            aligned_evts = align_eventmarkers_to_screen(evt_match, ts_match, log_evt, log_ts, screen_ts)

            # save events
            eventsfile = "-".join([project, subject, date, experiment, session]) + "_events" + align
            np.save(os.path.join(save_folder, eventsfile), aligned_evts)
            alignments.append(align)
    

    change_permissions_recursively(save_folder)


    return alignments

def find_logfiles(archive_folder):
    for (dirpath, dirnames, filenames) in os.walk(archive_folder):
        break

    logfiles = []

    # look for interesting log files
    for file in filenames:
        if file.endswith('_Cont.log'):
            if not file.endswith('Start_Cont.log'):
                logfiles.append(os.path.join(archive_folder, file))
    if len(logfiles) == 0: # empty
        raise ValueError('No logfiles found in archive_folder')

    return logfiles

def find_oe_recording(archive_folder):
    # Look for oe recording folder
    for (dirpath, dirnames, filenames) in os.walk(archive_folder):
        break

    oe_recording = None
    for directory in dirnames:
        if directory.startswith('Record'):
            oe_recording = archive_folder
    if oe_recording == None:
        for directory in dirnames:
            for (dirpath, dirnames2, filenames) in os.walk(os.path.join(archive_folder, directory)):
                break
            for directory2 in dirnames2:
                if directory2.startswith('Record'):
                    oe_recording = os.path.join(archive_folder, directory)
    if oe_recording == None:
        print(dirnames)
        raise ValueError('No open ephys recording folder found in archive_folder')

    return oe_recording

def format_iso8601(start_time, conversion="%H%M%S"):
    # Other useful conversion: "%Y%m%d"
    # our ISO 8601 strings always end in Z
    if start_time[-1] == 'Z':
        start_time = start_time[:-1]
    else:
        start_time = start_time

    start = datetime.fromisoformat(start_time)
    return start.strftime(conversion)

def find_subsequence(seq, subseq): # https://stackoverflow.com/questions/7100242/python-numpy-first-occurrence-of-subarray/20689091#20689091
    target = np.dot(subseq, subseq)
    candidates = np.where(np.correlate(seq,
                                    subseq, mode='valid') == target)[0]
    # some of the candidates entries may be false positives, double check
    check = candidates[:, np.newaxis] + np.arange(len(subseq))
    mask = np.all((np.take(seq, check) == subseq), axis=-1)
    return candidates[mask]
    

def align_eventmarkers_to_log(evt, ts, log_evt, log_ts, screen_ts):

    # find the start logfile eventmarker sequence [  992, random, random, random,   994,    10]
    start_sequence = log_evt[:6]
    st_log = find_subsequence(evt, start_sequence)
    
    if st_log.size == 1:
        st_log = st_log[0]
        n_match=log_evt.size

        events_match = evt[st_log:st_log+n_match]
        ts_match = ts[st_log:st_log+n_match]
        log_events_match = log_evt

        if events_match.size < log_events_match.size:
            print('Open Ephys recording ended too soon, replacing timestamps with nans')
            nmissing = log_events_match.size-events_match.size
            events_match = np.append(events_match, log_events_match[-nmissing:])
            ts_match = np.append(ts[st_log:st_log+n_match], np.full(nmissing, np.nan))
        
    elif st_log.size == 0:
        print('Start sequence not found in eventmarkers ', start_sequence)
        print('Assuming a late recording start, try to align to start of OE')
        start_sequence = evt[:100]
        st_evt = find_subsequence(log_evt, start_sequence)
        if st_evt.size != 1:
            print(start_sequence)
            raise ValueError('Automatic alignment of these OE events and logfile failed')
            
        st_evt = st_evt[0]
        n_match = log_evt.size - st_evt

        events_match = evt[:n_match]
        ts_match = ts[:n_match]
        log_events_match = log_evt
        
        if events_match.size < log_events_match.size:
            print('Open Ephys recording started too late, replacing timestamps with nans')
            nmissing = log_events_match.size-events_match.size
            events_match = np.append(log_events_match[:st_evt], events_match)
            ts_match = np.append(np.full(nmissing, np.nan), ts_match)
            
    elif st_log.size > 1:
        print(start_sequence)
        raise ValueError('Too many matches for this logfile, no unique log start eventmarker')

    if np.sum(events_match - log_events_match) != 0:
        raise ValueError('Open ephys and log events dont match for %i markers'%(n_match))
        
    return events_match, ts_match


def create_oe_eventmarkers(this_recording, fix_type='strobe'):
    # find events
    evt_info = this_recording.info['events'][0] # 0 is events, 1 is Message_Center
    print(evt_info)
    evt_file = os.path.join(this_recording.directory, 
                            "events",
                            evt_info['folder_name'], 
                            "full_words.npy")

    # convert and exclude eventmarkers
    evtPd = this_recording.events
    evt = np.load(evt_file).astype(int)
    ts = evtPd.timestamp.to_numpy()

    # If 16bit event-markers are used, combine 2 full words
    if evt_info['type'] == "int16":
        evt16 = np.zeros((evt.shape[0]), int)
        for irow in range(evt.shape[0]):
            evt16[irow] = int(format(evt[irow,1], "08b") + format(evt[irow,0], "08b"), 2)
        evt = evt16
        
    if evt.size != ts.size:
        err = "size of events in numpy file and pandas table do not match"
        raise ValueError(err)

    # use evt before the rising strobe bit to include the right eventmarkers
    strobe_rising = np.logical_and(evtPd.channel.to_numpy() == 16, evtPd.state.to_numpy() == 1)
    # nothing available before the first eventmarker
    strobe_rising[0] = False
    inc = np.nonzero(strobe_rising)[0] - 1

    if inc.size == 0:
        # file without strobe bit
        evt_ts = np.stack((evt, ts),axis=-1)
        unused, ind = np.unique(evt_ts, axis=0, return_index=True)
        inc = np.sort(ind)

    if np.any(evt[inc] > 2**15):
        wrong_idx = inc[evt[inc] > 2**15]
        print('Eventmarkers are wrong and include the strobe bit', evt[wrong_idx])
        if fix_type == 'strobe':
            print('Trying to replace with strobes - 2**15')
            evt = evt[inc + 1] - 2**15
            ts = ts[inc + 1]
            return evt, ts
        elif fix_type == 'shift':
            wrong_bool = evt[inc] > 2**15
            wrong_idx = inc[wrong_bool]
            print('Eventmarkers are wrong and include the strobe bit', evt[wrong_idx])
            # try one eventmarker earlier
            evt[wrong_idx] = evt[wrong_idx - 1]
            print('replaced with ', evt[wrong_idx])
            # check for weird starting high case
            if evt[wrong_idx[0]] > 2**15 and wrong_bool[0]:
                new = evt[wrong_idx[0]] - 2**15
                print('Special fix, replacing %i with %i'%(evt[wrong_idx],new))
                evt[wrong_idx[0]] = new
            if np.any(evt[wrong_idx] > 2**15):
                raise ValueError('Automatic fix failed')
        else:
            raise ValueError('Unrecognized fix_type %s'%(fix_type))
            
    evt = evt[inc]
    ts = ts[inc]

    return evt, ts

def align_eventmarkers_to_screen(evt, ts, log_evt, log_ts, screen_ts):
    # find log screen time for each eventmarker

    start_sess = 992
    end_sess = 993

    all_sts = np.where(evt == start_sess)[0]
    all_ends = np.where(evt == end_sess)[0]

    if all_ends.size == 0: # ended recording too soon
        if all_sts.size == 0:
            print('Started and ended recording too soon, using logfile to align')
            evt_match, ts_match = align_eventmarkers_to_log(evt, ts, log_evt, log_ts, screen_ts)
            return align_eventmarkers_to_screen(evt_match, ts_match, log_evt, log_ts, screen_ts)
        st = all_sts[0]
        end = -1
        end_sess = evt[-1]
    elif all_sts.size == 0: # started recording too late
        st = 0
        start_sess = evt[0]
        end = all_ends[0]
    elif all_ends[0] > all_sts[0]: # perfect case
        st = all_sts[0]
        end = all_ends[0]
    elif all_ends.size == 1 & all_sts.size == 1: 
        # ended recording too soon but started with an end OR started too late and ended with a start
        if all_ends[0] < len(evt)/2: # started with an end
            st = all_sts[0]
            end = -1
            end_sess = evt[end]
        else: # ended with a start
            st = 0
            start_sess = evt[st]
            end = all_ends[0]
    elif all_ends[1] > all_sts[0]: # usual case
        st = all_sts[0]
        end = all_ends[1]
    else:
        print('Too many starts and ends in eventmarkers, using logfile to align')
        evt_match, ts_match = align_eventmarkers_to_log(evt, ts, log_evt, log_ts, screen_ts)
        return align_eventmarkers_to_screen(evt_match, ts_match, log_evt, log_ts, screen_ts)

    # check that they match
    log_st = np.where(log_evt == start_sess)[0][0]
    log_end = np.where(log_evt == end_sess)[0][-1]

    if evt[st:end].size != log_evt[log_st:log_end].size:
        print('Eventmarkers in openephys and log length mismatch, using logfile to align')
        evt_match, ts_match = align_eventmarkers_to_log(evt, ts, log_evt, log_ts, screen_ts)
        return align_eventmarkers_to_screen(evt_match, ts_match, log_evt, log_ts, screen_ts)

    if np.sum(evt[st:end] - log_evt[log_st:log_end]) != 0:
        print(evt[st:end])
        print(log_evt[log_st:log_end])
        raise ValueError('Eventmarkers in openephys and log dont match')

    match_evt = evt[st:end]

    # align the times
    match_ts = ts[st:end]
    match_log_ts = screen_ts[log_st:log_end]

    # calculate the offset between sent and screen time in samples
    offset = (screen_ts[log_st:log_end] - log_ts[log_st:log_end])*evt_samplerate
    evt_screen_ts = match_ts + offset

    # original sample, screen sample, log screen times, eventmarkers 
    aligned_evts = np.column_stack((match_ts, evt_screen_ts, match_log_ts, match_evt))

    return aligned_evts


def load_aligned_eventmarkers(compute_folder, alignment, convert_int=True):
    for (dirpath, dirnames, filenames) in os.walk(compute_folder):
        break
    # load in events
    aligned_events = None
    for eventfile in filenames:
        if eventfile.split('.')[-2].endswith(alignment):
            if aligned_events is None:
                aligned_events = np.load(os.path.join(compute_folder, eventfile))
                if convert_int:
                    is_nan = np.isnan(np.sum(aligned_events, axis=1))
                    if np.any(is_nan):
                        print('Removing %i nan eventmarkers'%(np.count_nonzero(is_nan)))
                        print(aligned_events[is_nan,:])
                        aligned_events = aligned_events[np.logical_not(is_nan),:]
                    aligned_events = aligned_events.astype(int)
                basename = eventfile.split('_')[0]
            else:
                raise ValueError('Specified alignment %s matched multiple files'%(alignment))

    if aligned_events is None:
        raise ValueError('No aligned eventmarkers %s found, create first with snippet_eventmarkers'%(alignment))
        
    return aligned_events, basename

def find_all_alignments(compute_folder):
    evt_pattern = '*_events' + '[0-9]'*6 + '.npy'
    matches = glob.glob(os.path.join(compute_folder, evt_pattern))
    
    alignments = []
    for file in matches:
        head, tail = os.path.split(file)
        alignments.append(tail[-10:-4])

    if len(alignments) == 0:
        raise ValueError('No alignments found in folder %s'%(compute_folder))

    return alignments
