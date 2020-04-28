import sys
sys.path.append("..")
import numpy as np
from IceNet import dataloader, icecubeutils
from IceNet.datautils.pulsenormalizer import PulseNormalizer
from scipy import stats
from tqdm.notebook import tqdm


def read_files(file_list, labels):
    """
    Reads list of filenames using dataloader.get_pulses.

    Parameters
    ----------
    file_list: list of str
        Paths to simulation files.
    labels: list of str
        Labels to read from files.

    Returns
    -------
    x: list of array-like
        Events with pulses.
    y: list of array-like
        Truth labels of events.
    """
    x, y = [], []
    for files in file_list:
        xx, yy = dataloader.get_pulses(files, labels=labels)
        x.append(xx)
        y.append(yy)
    flatten = lambda superlist: [x for sublist in superlist for x in sublist]
    x = flatten(x)
    y = flatten(y)
    return x, y


def filter_pulses(x, y, max_pulses=None):
    """
    Filters out events with number of pulses over threshold.

    Parameters
    ----------
    x: list of array-like
        Container with feature pulses per event from dataloader.get_pulses.
    y: list of array-like
        Container with labels per event from dataloader.get_pulses.
    max_pulses: int
        Threshold over which to filter out pulses

    Returns
    -------
    x_filtered:
        Filtered events with pulses.
    y_filtered:
        Truth labels of filtered events.
    """
    if not max_pulses:
        return x, y

    pulse_mask = [i for i, event in enumerate(x) if len(event) < max_pulses]
    filter_pulses = lambda container: [container[i] for i in pulse_mask]
    x_filtered = filter_pulses(x)
    y_filtered = filter_pulses(y)
    return x_filtered, y_filtered


def polar2cart(phi, theta):
    return [
         np.sin(theta) * np.cos(phi),
         np.sin(theta) * np.sin(phi),
         np.cos(theta)
    ]


def cart2polar(x, y, z):
    phi = np.arctan2(y,z)
    phi += 2*np.pi if phi < 0 else 0
    return [phi,
           np.arctan2(np.sqrt(x**2 + y**2) , z)]


def get_light_distance(x, col_t=1):
    c = 299792458 # mps
    time_to_pos = [event[:,col_t]*1e-9*c for event in x]
    time_to_pos = [time.reshape(-1, 1) for time in time_to_pos]
    return time_to_pos


def get_dom_positions(x, col_domid=0):
    dom_pos_list = icecubeutils.get_dom_positions()
    hit_dom_pos = [dom_pos_list[event[:,0].astype(int)] for event in x]
    return hit_dom_pos


def get_ctxyz(x):
    # Variable name hit_doms is misleading, actually a pulse, since one DOM can have multiple hits; hit_doms implies unique DOM list
    # hit_doms = [dom_positions[event[:,0].astype(int)] for event in x]
    hit_dom_pos = get_dom_positions(x)
    time_to_pos = get_light_distance(x)
    all_pos = [np.concatenate((time, pos), axis=1) for time, pos in zip(time_to_pos, hit_dom_pos)]
    return all_pos


def calc_dist_matrix(event):
    npulses = len(event)
    idx = np.asarray(list(np.ndindex((npulses, npulses))))
    calc_dist = lambda i, j: np.linalg.norm(event[i] - event[j]) \
                            * np.sign(event[j, 0] - event[i, 0])

    dist_matrix = np.vectorize(calc_dist)(idx[:,0], idx[:,1])
    dist_matrix = dist_matrix.reshape((npulses, npulses))
    return dist_matrix


def calculate_weights(dist_matrices, mode='inverse_square'):
    # Make this more modular
    if mode == 'gauss':
        gauss = stats.norm(scale=100)
        weights = [gauss.pdf(m) / gauss.pdf(0) * np.sign(m) for m in tqdm(dist_matrices)]

    elif mode =='inverse_square':
        ### 1/d2 weights
        qq=1e-3
        weights = [1 / (1 + qq * m**2) * np.sign(m) for m in tqdm(dist_matrices)]

    return weights



def get_weight_matrices(x, mode='inverse_square'):
    all_pos = get_ctxyz(x)
    dists = [calc_dist_matrix(event) for event in tqdm(all_pos)]
    weights = calculate_weights(dists, mode)
    return weights


def get_nn_features(x):
    hit_doms = get_dom_positions(x)
    features = [np.concatenate((np.log10(event[:,2].reshape(-1,1)), # log charges
                                doms, # xyz
                                event[:,1].reshape(-1,1) # Time
                               ), axis=1)
                for event, doms in zip(x, hit_doms)]

    # Normalize
    pn = PulseNormalizer(features)
    features_normalized = pn.normalize(mode='gauss')

    return features_normalized


def get_edge_info(weight_matrix):
    n_pulses = weight_matrix.shape[0]
    idx = np.asarray(list(np.ndindex((n_pulses, n_pulses))))
    edge_weights = np.asarray([weight_matrix[i,j] for i,j in idx])
    idx = np.swapaxes(idx, 0, 1)
    return idx, edge_weights
