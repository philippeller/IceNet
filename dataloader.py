'''Read in IceTray hdf5 file and format correctly'''
from __future__ import division, print_function
import numpy as np
import h5py

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
    
    if sparse:
        import tensorflow as tf
        indices = []
        values = []
        dense_shape = [N_events, N_channels, N_pulses, N_features]
    
    else:
        X = np.zeros((N_events, N_channels, N_pulses, N_features), dtype=dtype)
    
    data_idx = 0
    bincount = np.bincount(pulses['Event'])
    
    # fill array
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
                
                if sparse:
                    indices.append([data_idx, channel_idx, hit_idx, i])
                    values.append(dtype(hit[feature]))
                else:
                    X[data_idx, channel_idx, hit_idx, i] = hit[feature]

        data_idx += 1       
    
    if sparse:
        X = tf.sparse.SparseTensor(indices, values, dense_shape)
    
        indices_y = []
        values_y = []
        dense_shape_y = [N_events, len(labels)]
    
        for data_idx in range(N_events):
            for i, label in enumerate(labels):
                indices_y.append([data_idx, i])
                values_y.append(dtype(truth[data_idx][label]))
        
        y = tf.sparse.SparseTensor(indices_y, values_y, dense_shape_y)
    
    else:
        y = np.empty((N_events, len(labels)), dtype=dtype)
        
        for i, label in enumerate(labels):
            y[:, i] = truth[:N_events][label]
    
    return X, y