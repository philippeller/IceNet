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
        N_events = truth.shape[0]
    
         
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
    
    data_idx = 0
    bincount = np.bincount(pulses['Event'])
    
    # fill array
    with tqdm(total=N_events) as pbar:
        for event_idx, num_pulses in enumerate(bincount):
            if num_pulses == 0:
                continue


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
    
    y = np.empty((N_events, len(labels)), dtype=dtype)

    for i, label in enumerate(labels):
        y[:, i] = truth[:N_events][label]

    return X, y


#@jit
def get_data_3d(
    fname,
    truth_i3key='MCInIcePrimary',
    pulses_i3key='SRTTWOfflinePulsesDC',
    features=["np.sum(p['charge'])"],
    labels=['z'],
    N_events=None, 
    dtype=np.float32,
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
    
    h = h5py.File(fname, 'r')

    truth = np.array(h[truth_i3key])
    pulses = np.array(h[pulses_i3key])

    if N_events is None:
        N_events = truth.shape[0]
    
    N_channels = 5160
    N_features = len(features)

    X = np.zeros((N_events, N_channels, N_features), dtype=dtype)
    
    data_idx = 0
    bincount = np.bincount(pulses['Event'])
    
    # fill array
    with tqdm(total=N_events) as pbar:
        for event_idx, num_pulses in enumerate(bincount):
            if num_pulses == 0:
                continue

            event_p = pulses[pulses['Event'] == event_idx]            
            # time conversion

            min_time = np.min(event_p['time'])
            max_time = np.max(event_p['time'])
            ave_time = np.mean(event_p['time'])

            event_p['time'] -= ave_time
            if max_time > min_time:
                event_p['time'] /= (max_time - min_time)

            for string_idx in range(86):
                string_p = event_p[event_p['string'] == string_idx + 1]\

                if len(string_p) == 0:
                    continue

                for dom_idx in range(60):
                    p = string_p[string_p['om'] == dom_idx + 1]

                    if len(p) == 0:
                        continue

                    channel_idx = 60 * string_idx + dom_idx

                    for i, feature in enumerate(features):
                        exec("X[data_idx, channel_idx, i] = %s"%feature)

            data_idx += 1
            pbar.update(1)

            if data_idx == N_events:
                break
    
    y = np.empty((N_events, len(labels)), dtype=dtype)

    for i, label in enumerate(labels):
        y[:, i] = truth[:N_events][label]

    return X, y
