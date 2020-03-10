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
        self.events = events
        self.norm_cols = norm_cols
        self.linear_parameters = self._get_linear_parameters()
        
    
    def _get_min_max(self):
        col_mins = [[] for i in range(len(self.norm_cols))]
        col_maxs = [[] for i in range(len(self.norm_cols))]
        
        for event in self.events:
            for i, col in enumerate(self.norm_cols): 
                col_mins[i].append(np.min(event[:, col]))
                col_maxs[i].append(np.max(event[:, col]))
        
        minmax = [[np.min(mins), np.max(maxs)] for mins, maxs in zip(col_mins, col_maxs)]
        
        return minmax
    
    
    def _get_linear_parameters(self):
        minmaxs = self._get_min_max()
        tminmax, qminmax = self._get_min_max()
        pars = [[mm[1]-mm[0], mm[0]] for mm in minmaxs]
        return np.asarray(pars)
    
    
    def _get_normalized_event(self, event):
        new_event = np.copy(event)
        features = new_event[:, self.norm_cols]
        norm_feat = (features - self.linear_parameters[:,1]) / self.linear_parameters[:,0]
        new_event[:, self.norm_cols] = norm_feat
        return new_event

    
    def normalize(self):
        nn = [self._get_normalized_event(event) for event in tqdm(self.events)]
        return nn
