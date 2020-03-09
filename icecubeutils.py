import numpy as np
import pickle
from tqdm.notebook import tqdm

ndoms = 5160


def get_doms_by_strings(fname='../../GeoCalibDetectorStatus_AVG_55697-57531_PASS2_SPE_withScaledNoise.pkl'):
    gcd_file = pickle.load(open(fname, 'rb'), encoding='latin1')

    dom_positions = gcd_file['geo']
    return dom_positions


def get_dom_positions():
    dom_positions = get_doms_by_strings()

    dom_pos_flat = []
    for string in dom_positions:
        for xyz in string:
            dom_pos_flat.append(xyz)
    dom_pos_flat = np.asarray(dom_pos_flat)
    return dom_pos_flat


def calculate_dom_distance_matrix():
    dom_pos_flat = get_dom_positions()
    dists = np.empty((ndoms, ndoms))
    for i, j in tqdm(np.ndindex(dists.shape), total = ndoms**2):
        dists[i, j] = (np.linalg.norm(dom_pos_flat[i] - dom_pos_flat[j]))
    return dists


def normalize_dom_distance_matrix(dists):
    std = np.std(dists)
    return dists / 100.

def get_dom_distance_matrix(cache_name='./dom_dist_matrix.npy'):
    try:
        dists = np.load(cache_name)
    except:
        dists = calculate_dom_distance_matrix()
        np.save(cache_name, dists)
    dists = normalize_dom_distance_matrix(dists)
    return dists


def get_n_closest_doms(dom, n, dist_matrix):
    dists = dist_matrix[dom]
    order = np.argsort(dists)
    dists_ordered = dists[order]
    doms = order[:n]
    return doms



