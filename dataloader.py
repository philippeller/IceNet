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
    #ragged_left=True,
    sparse=True,
    ragged=False,
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
    ragged_left : bool
        align hits to the right end of tensor
    sparse : bool
        if true, retrun feature vector as tf.sparse.SparseTensor
    ragged : bool
        if true, return feature vector as tf.RaggedTensor
        
    Returns:
    --------
    
    X : array
        feature array of shape (N_events, N_channels, N_pulses, N_features)
    y : array
        label array of shape (N_events, N_labels)
    
    '''
    
    assert not (sparse and ragged)

    h = h5py.File(fname, 'r')

    truth = np.array(h[truth_i3key])
    pulses = np.array(h[pulses_i3key])

    if N_events is None:
        nevents = lambda x: len(np.unique(x['Event'])) # Get number of unique events in container
        N_events = min(nevents(pulses), nevents(truth))
    
         
    # need to figure out max number of pulses in any event / string / dom first to allocate array
    max_pulses = np.int(np.max(pulses['vector_index']) + 1)
    
    print('max pulses = ', max_pulses)
    
    N_channels = 5160
    N_pulses = max_pulses
    N_features = len(features)
    
    if sparse:
        import tensorflow as tf
        indices = []
        values = []
        dense_shape = [N_events, N_channels, N_pulses, N_features]
    
    elif ragged:
        import tensorflow as tf
        X = []
    
    else:
        X = np.zeros((N_events, N_channels, N_pulses, N_features), dtype=dtype)

    y = np.zeros((N_events, len(labels)), dtype=dtype)
    
    data_idx = 0
    bincount = np.bincount(pulses['Event'])
    
    # fill array
    with tqdm(total=N_events) as pbar:
        for event_idx, num_pulses in enumerate(bincount):
            if num_pulses == 0:
                continue

            l = truth[truth['Event'] == event_idx]
            if not l:
                continue
            for i, label in enumerate(labels):
                y[data_idx, i] = l[label]

            if ragged:
                X.append([[[dtype(0.) for i in range(len(features))]] for i in range(N_channels)])

            p = pulses[pulses['Event'] == event_idx]

            for hit in p:

                hit_idx = hit['vector_index']
                string_idx = hit['string'] - 1
                dom_idx = hit['om'] - 1
                channel_idx = 60 * string_idx + dom_idx

                if ragged:
                    X[-1][channel_idx].append([])

                for i, feature in enumerate(features):

                    if sparse:
                        indices.append([data_idx, channel_idx, hit_idx, i])
                        values.append(dtype(hit[feature]))
                    elif ragged:
                        X[-1][channel_idx][-1].append(dtype(hit[feature]))
                    else:
                        X[data_idx, channel_idx, hit_idx, i] = hit[feature]

            data_idx += 1
            pbar.update(1)
            
            if data_idx == N_events:
                break
        
    if ragged:
        X = tf.ragged.constant(X)
                                                                             
    if sparse:
        X = tf.sparse.SparseTensor(indices, values, dense_shape)

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
    N_features = 6 #len(features)

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
                    #X[data_idx, channel_idx, 0] = len(p)
                    X[data_idx, channel_idx, 0] = np.sum(p['charge'])
                    # do charge weighted percentiles
                    X[data_idx, channel_idx, 1:] = np.percentile(p['time'], [0, 25, 50, 75, 100])

            data_idx += 1
            pbar.update(1)

            if data_idx == N_events:
                break  
    
    return X, y, r



def get_single_hits(
    fname,
    geo='geo_array.npy',
    truth_i3key='MCInIcePrimary',
    pulses_i3key='SRTTWOfflinePulsesDC',
    labels=['x', 'y', 'z', 'time', 'zenith', 'azimuth', 'energy'],
    N_hits=None, 
    dtype=np.float32,
    ):
    '''Load in icetray hdf file for machine learning
    
    Parameters:
    -----------
    fname : str
        filename / path
    geo : str
        filename / path to npy geometry file
    truth_i3key : str
        key of truth information
    pulses_i3key : str
        pulse series
    labels : list
        labels for training vector
    N_hits : int (optional)
        number of hits to read
    dtype : dtype
        dtype of output arrays
        
    Returns:
    --------
    
    X : array
        feature array of shape (N_hits, 4)
    w : array
        weights of hits (= charge)
    y : array
        label array of shape (N_hits, N_labels)
    
    '''
    
    h = h5py.File(fname, 'r')

    truth = np.array(h[truth_i3key])
    pulses = np.array(h[pulses_i3key])

    geo = np.load(geo)

    if N_hits is None:
        N_hits = len(pulses) 
    
    X = np.zeros((N_hits, 4), dtype=dtype)
    w = np.zeros((N_hits,), dtype=dtype)
    y = np.zeros((N_hits, len(labels)), dtype=dtype)
    
    
    data_idx = 0
    bincount = np.bincount(pulses['Event'])
    
    # fill array
    with tqdm(total=N_hits) as pbar:
        for event_idx, num_pulses in enumerate(bincount):
            if num_pulses == 0:
                continue

            l = truth[truth['Event'] == event_idx]
            if not l:
                continue

            last_idx = min(data_idx+num_pulses, N_hits)

            for i, label in enumerate(labels):
                y[data_idx:last_idx, i] = l[label]

            p = pulses[pulses['Event'] == event_idx]

            # Vectorize me!
            for hit in p:

                pbar.update(1)

                if data_idx == N_hits:

                    return X, w, y

                string_idx = hit['string'] - 1
                dom_idx = hit['om'] - 1

                X[data_idx, 0:3] = geo[string_idx, dom_idx]
                X[data_idx, 3] = hit['time']
                w[data_idx] = hit['charge']

                data_idx += 1

    return X, w, y

def get_event_hits(
    fname,
    geo='geo_array.npy',
    truth_i3key='MCInIcePrimary',
    pulses_i3key='SRTTWOfflinePulsesDC',
    labels=['x', 'y', 'z', 'time', 'zenith', 'azimuth', 'energy'],
    N_events=None, 
    dtype=np.float32,
    ):
    '''Load in icetray hdf file for machine learning
    
    Parameters:
    -----------
    fname : str
        filename / path
    geo : str
        filename / path to npy geometry file
    truth_i3key : str
        key of truth information
    pulses_i3key : str
        pulse series
    labels : list
        labels for training vector
    N_events : int (optional)
        number of events to read
    dtype : dtype
        dtype of output arrays
        
    Returns:
    --------
    
    X : list of arrays
        feature array of shape (N_hits, 4)
    w : list of arrays
        weights of hits (= charge) (N_hits,)
    y : list of arrays
        label array of shape (N_labels,)
    
    '''
    
    h = h5py.File(fname, 'r')

    truth = np.array(h[truth_i3key])
    pulses = np.array(h[pulses_i3key])

    geo = np.load(geo)

    if N_events is None:
        nevents = lambda x: len(np.unique(x['Event'])) # Get number of unique events in container
        N_events = min(nevents(pulses), nevents(truth))
    
    data_idx = 0
    bincount = np.bincount(pulses['Event'])
    
    Xs = []
    ws = []
    ys = []

    # fill array
    with tqdm(total=N_events) as pbar:
        for event_idx, num_pulses in enumerate(bincount):
            if num_pulses == 0:
                continue

            l = truth[truth['Event'] == event_idx]
            if not l:
                continue

            X = np.zeros((num_pulses, 4), dtype=dtype)
            w = np.zeros((num_pulses,), dtype=dtype)
            y = np.zeros((len(labels),), dtype=dtype)

            for i, label in enumerate(labels):
                y[i] = l[label]

            p = pulses[pulses['Event'] == event_idx]

            # Vectorize me!
            for i, hit in enumerate(p):
                string_idx = hit['string'] - 1
                dom_idx = hit['om'] - 1

                X[i, 0:3] = geo[string_idx, dom_idx]
                X[i, 3] = hit['time']
                w[i] = hit['charge']

            Xs.append(X)
            ws.append(w)
            ys.append(y)

            data_idx += 1
            pbar.update(1)

            if data_idx == N_events:
                return Xs, ws, ys


    return X, w, y

def get_event_charge(
    fname,
    truth_i3key='MCInIcePrimary',
    pulses_i3key='SRTTWOfflinePulsesDC',
    labels=['x', 'y', 'z', 'time', 'zenith', 'azimuth', 'energy'],
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
    labels : list
        labels for training vector
    N_events : int (optional)
        number of events to read
    dtype : dtype
        dtype of output arrays
        
    Returns:
    --------
    
    X : array
        feature array of shape (N_events,)
    y : arrays
        label array of shape (N_events, N_labels,)
    
    '''
    
    h = h5py.File(fname, 'r')

    truth = np.array(h[truth_i3key])
    pulses = np.array(h[pulses_i3key])


    if N_events is None:
        nevents = lambda x: len(np.unique(x['Event'])) # Get number of unique events in container
        N_events = min(nevents(pulses), nevents(truth))

    N_labels = len(labels)
    
    data_idx = 0
    bincount = np.bincount(pulses['Event'])

    X = np.empty((N_events,), dtype=dtype)
    y = np.empty((N_events, N_labels), dtype=dtype)

    # fill array
    with tqdm(total=N_events) as pbar:
        for event_idx, num_pulses in enumerate(bincount):
            if num_pulses == 0:
                continue

            l = truth[truth['Event'] == event_idx]
            if not l:
                continue

            for i, label in enumerate(labels):
                y[data_idx, i] = l[label]

            p = pulses[pulses['Event'] == event_idx]

            X[data_idx] = np.sum(p['charge'])

            data_idx += 1
            pbar.update(1)

            if data_idx == N_events:
                return X, y

    return X, y
