'''Read in IceTray hdf5 file and format correctly'''
from __future__ import division, print_function
import numpy as np
import h5py
from numba import jit
from tqdm import tqdm

def get_data(
    fname,
    truth_i3key='MCInIcePrimary',
    pulses_i3key='SRTTWOfflinePulsesDC',
    features=['time', 'charge'],
    labels=['zenith', 'azimuth'],
    N_events=None, 
    dtype=np.float32,
    ):
    '''Load in icetray hdf file for machine learning
    
    Parameters:
    -----------
    fname : str
        filename / path
    truth_i3key : str
        key of truth information
    pulses_i3key : str
        pulse series
    features : list
        features for training vector
    labels : list
        labels for training vector
    N_events : int (optional)
        number of events to read
    dtype : dtype
        dtype of output arrays
        
    Returns:
    --------
    
    X : array
        feature array of shape (N_events, N_channels, N_pulses, N_features)
    y : array
        label array of shape (N_events, N_labels)
    
    '''
    
    h = h5py.File(fname, 'r')

    truth = np.array(h[truth_i3key])
    pulses = np.array(h[pulses_i3key])

    if N_events is None:
        N_events = truth.shape[0]
    
         
    # need to figure out max number of pulses in any event / string / dom first to allocate array
    max_pulses = np.int(np.max(pulses['vector_index']) + 1)
    
    print('max pulses = ', max_pulses)
    
    N_channels = 5160
    N_pulses = max_pulses
    N_features = len(features)
    
    X = np.zeros((N_events, N_channels, N_pulses, N_features), dtype=dtype)
    
    data_idx = 0
    bincount = np.bincount(pulses['Event'])
    
    # fill array
    with tqdm(total=N_events) as pbar:
        for event_idx, num_pulses in enumerate(bincount):
            if num_pulses == 0:
                continue

            p = pulses[pulses['Event'] == event_idx]

            for hit in p:

                hit_idx = hit['vector_index']
                string_idx = hit['string'] - 1
                dom_idx = hit['om'] - 1
                channel_idx = 60 * string_idx + dom_idx

                for i, feature in enumerate(features):

                    X[data_idx, channel_idx, hit_idx, i] = hit[feature]

            data_idx += 1
            pbar.update(1)
            
            if data_idx == N_events:
                break
        
    
    y = np.empty((N_events, len(labels)), dtype=dtype)

    for i, label in enumerate(labels):
        y[:, i] = truth[:N_events][label]

    return X, y


def get_data_3d(
    fname,
    truth_i3key='MCInIcePrimary',
    pulses_i3key='SRTTWOfflinePulsesDC',
    reco_i3key='SPEFit2_DC',
    labels=['z'],
    reco_labels=['z'],
    N_events=None, 
    dtype=np.float32,
    min_pulses=8,
    ):
    '''Load in icetray hdf file for machine learning
    This is in 3d format = summary of pulses per DOM instead of pulses
    
    Parameters:
    -----------
    fname : str
        filename / path
    truth_i3key : str
        key of truth information
    pulses_i3key : str
        pulse series
    #features : list
    #    features for training vector
    #labels : list
    #    labels for training vector
    N_events : int (optional)
        number of events to read
    dtype : dtype
        dtype of output arrays
    min_pulses : int
        minimum number of pulses per event
    Returns:
    --------
    
    X : array
        feature array of shape (N_events, N_channels, N_pulses, N_features)
    y : array
        label array of shape (N_events, N_labels)
    r :  array
        reco array of shape (N_events, N_reco_labels)
    
    '''
    
    h = h5py.File(fname, 'r')
    
    truth_event_idx = np.array(h[truth_i3key]['Event'])
    pulses_event_idx = np.array(h[pulses_i3key]['Event'])
    reco_event_idx = np.array(h[reco_i3key]['Event'])
    
    
    load_labels = list(set(labels).difference(set(['dir_x', 'dir_y', 'dir_z'])))
    reco_load_labels = list(set(reco_labels).difference(set(['dir_x', 'dir_y', 'dir_z'])))
    
    truth = np.asarray(h[truth_i3key])[load_labels]
    pulses = np.asarray(h[pulses_i3key])[['string', 'om', 'charge', 'time']]
    reco = np.asarray(h[reco_i3key])[reco_load_labels]

    
    if 'dir_x' in labels:
        
        from numpy.lib import recfunctions
        
        dir_truth = np.zeros(truth.shape, dtype=[('dir_x', dtype), ('dir_y', dtype), ('dir_z', dtype)])
        dir_reco = np.zeros(truth.shape, dtype=[('dir_x', dtype), ('dir_y', dtype), ('dir_z', dtype)])
    
        dir_truth['dir_x'] = np.sin(truth['zenith']) * np.cos(truth['azimuth'])
        dir_truth['dir_y'] = np.sin(truth['zenith']) * np.sin(truth['azimuth'])
        dir_truth['dir_z'] = np.cos(truth['zenith'])
        
        dir_reco['dir_x'] = np.sin(reco['zenith']) * np.cos(reco['azimuth'])
        dir_reco['dir_y'] = np.sin(reco['zenith']) * np.sin(reco['azimuth'])
        dir_reco['dir_z'] = np.cos(reco['zenith'])
        
        
        truth = recfunctions.merge_arrays([truth, dir_truth], flatten=True)
        reco = recfunctions.merge_arrays([reco, dir_reco], flatten=True)

    pulses['string'] -= 1
    pulses['om'] -= 1

    N_channels = 5160
    N_features = 7 #len(features)

    # ToDo
    
    bincount = np.bincount(pulses_event_idx)

    N_events_total = np.sum(bincount >= min_pulses)

    if N_events is None:
        N_events = N_events_total
    else:
        assert N_events <= N_events_total

           
    X = np.zeros((N_events, N_channels, N_features), dtype=dtype)
    y = np.zeros((N_events, len(labels)), dtype=dtype)
    r = np.zeros((N_events, len(labels)), dtype=dtype)

  

    data_idx = 0
    
    with tqdm(total=N_events) as pbar:

    
        for event_idx in np.where(bincount >= min_pulses)[0]:

            # fill truth info
            this_truth = truth[truth_event_idx == event_idx]
            this_reco = reco[reco_event_idx == event_idx]
            event_p = pulses[pulses_event_idx == event_idx]            

            # time conversion
            # shift by median and divide by 1000
            shift = np.median(event_p['time'])
            if 'time' in labels:
                this_truth['time'] -= shift
                this_truth['time'] /= 1e3
            if 'time' in reco_labels:
                this_reco['time'] -= shift
                this_reco['time'] /= 1e3 
            event_p['time'] -= shift
            event_p['time'] /= 1e3

            for i, label in enumerate(labels):
                y[data_idx, i] = this_truth[label]
            if len(this_reco) > 0:
                for i, label in enumerate(reco_labels):
                    r[data_idx, i] = this_reco[label]



            stringcount = np.bincount(event_p['string'])

            for string_idx in np.where(stringcount > 0)[0]:

                string_p = event_p[event_p['string'] == string_idx]
                omcount = np.bincount(string_p['om'])

                for dom_idx in np.where(omcount > 0)[0]:

                    if omcount[dom_idx] == 0:
                        continue

                    p = string_p[string_p['om'] == dom_idx]

                    channel_idx = 60 * string_idx + dom_idx

                    #X[data_idx, channel_idx, 0] = 1
                    X[data_idx, channel_idx, 0] = len(p)
                    X[data_idx, channel_idx, 1] = np.sum(p['charge'])
                    X[data_idx, channel_idx, 2:] = np.percentile(p['time'], [0, 30, 50, 70, 100])

            data_idx += 1
            pbar.update(1)

            if data_idx == N_events:
                break  
    
    return X, y, r


def get_pulses(
    fname,
    truth_i3key='MCInIcePrimary',
    pulses_i3key='SRTTWOfflinePulsesDC',
    features=['time', 'charge'],
    labels=['zenith', 'azimuth'],
    N_events=None, 
    dtype=np.float32,
    ):


    h = h5py.File(fname, 'r')

    truth = np.array(h[truth_i3key])
    pulses = np.array(h[pulses_i3key])

    if N_events is None:
        N_events = truth.shape[0]
    
    x = []
    y = []
    
    data_idx = 0
    bincount = np.bincount(pulses['Event'])
    
    # fill array
    
    for event_idx, num_pulses in tqdm(enumerate(bincount), total=len(bincount)):
        if num_pulses == 0:
            continue
        
        p = pulses[pulses['Event'] == event_idx]
        hitlist = []
        for hit in p:
            
            hit_idx = hit['vector_index']
            string_idx = hit['string'] - 1
            dom_idx = hit['om'] - 1
            channel_idx = 60 * string_idx + dom_idx
            
            feature_vector = [channel_idx]
            for i, feature in enumerate(features):
                if (feature == 'time'):
                    time_axis = i+1
                feature_vector.append(hit[feature])
            
            hitlist.append(feature_vector)
            
        # Sort pulses by time for each event
        if ('time' in features):
            hitlist = np.asarray(hitlist)
            time_idx = np.argsort(hitlist[:, time_axis])
            hitlist = hitlist[time_idx]
        
        x.append(hitlist)
        
        feature = np.empty(len(labels), dtype=dtype)


        for i, label in enumerate(labels):
            feature[i] = truth[data_idx][label]
        y.append(feature)
        
        data_idx += 1       
    
    return x, y
