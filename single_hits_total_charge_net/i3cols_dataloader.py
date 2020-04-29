import os
import numpy as np

def get_energies(mcprimary, mctree, mctree_idx, dtype=np.float32):
    '''Get energies per event'''
    
    neutrino_energy = mcprimary['energy']
    track_energy = np.zeros_like(neutrino_energy, dtype=dtype)
    invisible_energy = np.zeros_like(neutrino_energy, dtype=dtype)
    
    for i in range(len(mctree_idx)):
        this_idx = mctree_idx[i]
        this_mctree = mctree[this_idx['start'] : this_idx['stop']]
        pdg = this_mctree['particle']['pdg_encoding']
        en = this_mctree['particle']['energy']
    
        muon_mask = np.abs(pdg) == 13
        if np.any(muon_mask):
            track_energy[i] = np.max(en[muon_mask])

        invisible_mask = (np.abs(pdg) == 12) | (np.abs(pdg) == 14) | (np.abs(pdg) == 16) 
        # exclude primary:
        invisible_mask[0] = False
        if np.any(invisible_mask):
            # we'll make the bold assumptions that none of the neutrinos re-interact ;)
            invisible_energy[i] = np.sum(en[invisible_mask])

    cascade_energy = neutrino_energy - track_energy - invisible_energy
    return neutrino_energy, track_energy, cascade_energy

def get_total_charge(hits, hits_idx, dtype=np.float32):
    '''Get charge per event'''
    
    total_charge = np.zeros(hits_idx.shape, dtype=dtype)
    
    for i in range(len(hits_idx)):
        this_idx = hits_idx[i]
        this_hits = hits[this_idx['start'] : this_idx['stop']]

        total_charge[i] = np.sum(this_hits['pulse']['charge'])
        
    return total_charge

def load_data(dir='/home/iwsatlas1/peller/work/oscNext/level7_v01.04/140000_i3cols',
              labels=['x', 'y', 'z', 'time', 'azimuth','zenith', 'cascade_energy', 'track_energy'],
              geo='geo_array.npy',
              dtype=np.float32):
    """
    Create training data for hit and charge net
    
    Returns:
    --------
    single_hits : ndarray
        shape (N_hits, 5)
    repeated_params : ndarray
        shape (N_hits, len(labels))
    total_charge : ndarray
        shape (N_events,)
    params : ndarray
        shape (N_events, len(labels))
    """
    
    hits_idx = np.load(os.path.join(dir, 'SRTTWOfflinePulsesDC/index.npy'))
    hits = np.load(os.path.join(dir, 'SRTTWOfflinePulsesDC/data.npy'))
    mctree_idx = np.load(os.path.join(dir, 'I3MCTree/index.npy'))
    mctree = np.load(os.path.join(dir, 'I3MCTree/data.npy'))
    mcprimary = np.load(os.path.join(dir, 'MCInIcePrimary/data.npy'))

    geo = np.load(geo)
    
    # constrcut hits array

    # shape N x (x, y, z, t, q)
    single_hits = np.empty(hits.shape + (5,), dtype=dtype)
    string_idx = hits['key']['string'] - 1
    om_idx = hits['key']['om'] - 1

    single_hits[:, 0:3] = geo[string_idx, om_idx]
    single_hits[:, 3] = hits['pulse']['time']
    single_hits[:, 4] = hits['pulse']['charge']
    
    total_charge = get_total_charge(hits, hits_idx, dtype=dtype)
    
    # construct params array

    neutrino_energy, track_energy, cascade_energy = get_energies(mcprimary, mctree, mctree_idx, dtype=dtype)
    
    params = np.empty(mcprimary.shape + (len(labels), ), dtype=dtype)

    for i, label in enumerate(labels):
        if label == 'x': params[:, i] = mcprimary['pos']['x']
        elif label == 'y': params[:, i] = mcprimary['pos']['y']
        elif label == 'z': params[:, i] = mcprimary['pos']['z']
        elif label == 'time': params[:, i] = mcprimary['time']
        elif label == 'azimuth': params[:, i] = mcprimary['dir']['azimuth']
        elif label == 'zenith': params[:, i] = mcprimary['dir']['zenith']
        elif label == 'neutrino_energy': params[:, i] = neutrino_energy
        elif label == 'energy': params[:, i] = track_energy + cascade_energy
        elif label == 'cascade_energy': params[:, i] = cascade_energy
        elif label == 'track_energy': params[:, i] = track_energy

    repeats = (hits_idx['stop'] - hits_idx['start']).astype(np.int64)
    repeated_params = np.repeat(params, repeats=repeats, axis=0)

    
    return single_hits, repeated_params, total_charge, params, labels

def load_events(dir='/home/iwsatlas1/peller/work/oscNext/level7_v01.04/140000_i3cols',
              labels=['x', 'y', 'z', 'time', 'azimuth','zenith', 'cascade_energy', 'track_energy'],
              geo='geo_array.npy',
              dtype=np.float32):
    """
    Create event=by=event data for hit and charge net
    
    Returns:
    --------
    list of:
        single_hits : ndarray
            shape (N_hits, 5)
        total_charge : float
        params : ndarray
            shape (len(labels))
    labels
    """
    
    hits_idx = np.load(os.path.join(dir, 'SRTTWOfflinePulsesDC/index.npy'))
    
    single_hits, repeated_params, total_charge, params, labels = load_data(dir=dir, labels=labels, geo=geo, dtype=dtype)
    
    events = []
    
    for i in range(len(total_charge)):
        event = {}
        event['total_charge'] = total_charge[i]
        event['hits'] = single_hits[hits_idx[i]['start'] : hits_idx[i]['stop']]
        event['params'] = params[i]
        events.append(event)
    return events, labels