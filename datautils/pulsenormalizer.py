import numpy as np
from tqdm.notebook import tqdm

class PulseNormalizer:
    '''
    Takes pulses from dataloader.get_pulses and processes them for NN input
    Minmax normalizer
    
    Parameters:
    ----------
    events: list
        List with arrays of pulse information
    norm_cols: list
        Indices of columns to normalize
    '''
    def __init__(self, events, norm_cols):
        self._events = events
        self._norm_cols = norm_cols if norm_cols else list(range(events[0].shape[1]))

        self.mode = None
        self._linear_parameters = self._get_linear_parameters()
        self._norm_parameters = self._get_norm_parameters()


    def _get_min_max(self):
        col_mins = [[] for i in range(len(self._norm_cols))]
        col_maxs = [[] for i in range(len(self._norm_cols))]

        for event in self._events:
            for i, col in enumerate(self._norm_cols):
                col_mins[i].append(np.min(event[:, col]))
                col_maxs[i].append(np.max(event[:, col]))

        minmax = [[np.min(mins), np.max(maxs)] for mins, maxs in zip(col_mins, col_maxs)]

        return minmax


    def _get_linear_parameters(self):
        minmaxs = self._get_min_max()
        pars = [[mm[1]-mm[0], mm[0]] for mm in minmaxs]
        return np.asarray(pars)


    def _get_norm_parameters(self):
        norm_parameters = []
        for col in self._norm_cols:
            features = np.concatenate([ev[:, col] for ev in self._events])
            pars = [np.mean(features), np.std(features)]
            norm_parameters.append(pars)
        return np.asarray(norm_parameters)


    def _normalize_event(self, features):
        if self.mode == 'minmax':
            return (features - self._linear_parameters[:,1]) / self._linear_parameters[:,0]
        elif self.mode == 'gauss':
            return (features - self._norm_parameters[:, 0]) / self._norm_parameters[:,1]

    def _get_normalized_event(self, event):
        new_event = np.copy(event)
        features = new_event[:, self._norm_cols]
        norm_feat = self._normalize_event(features)
        new_event[:, self._norm_cols] = norm_feat
        return new_event


    def normalize(self, mode):
        self.mode = mode
        nn = [self._get_normalized_event(event) for event in tqdm(self._events)]
        return nn
