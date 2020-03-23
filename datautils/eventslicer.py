import numpy as np
from tqdm.notebook import tqdm


class EventSlicer:
    '''
    Takes events from dataloader.get_pulses and puts them in temporal slices
    Used for RNNs
    
    Parameters:
    ----------
    x: list
        List of pulses
    pads: float
        Number to fill non-events with
    '''
    def __init__(self, x, pads=-1):
        self.x = x
        self.time_windows = self._get_time_windows()
        min_times = self.time_windows[:, 0]
        max_times = self.time_windows[:, 1]
        self.max_duration = np.max(max_times - min_times)
        self.ndoms = 5160
        self.pads = pads
        self.padded = None
    
    
    def _get_time_windows(self):
        x = self.x
        min_times = []
        max_times = []
        for event in tqdm(x):
            hit_times = event[:, 1]
            charges = event[:, 2]
            hit_times = hit_times[charges!=0]
            if hit_times.size == 0: # Catch events with no pulses
                tmin = 0
                tmax = 0
            else:
                tmin = np.min(hit_times)
                tmax = np.max(hit_times)
            min_times.append(tmin)
            max_times.append(tmax)
    
        min_times = np.asarray(min_times)
        max_times = np.asarray(max_times)
    
        time_windows = np.asarray([min_times, max_times]).T
        
        return time_windows
    
    
    def _get_linear_parameters(self, ntokens):
        m = (self.max_duration) / (ntokens-1)
        return m
    
    
    def _get_hit_slices(self, event, ntokens, nspread):
        '''
        Converts hit times into tokenized indices
        '''
        hit_times = event[:,1]
#         min_time = np.min(hit_times)
#         hit_times = hit_times - min_time
        d = self.max_duration - np.max(hit_times)
        hit_times = hit_times + d
        # If padded, reserve spots for spread values
        if self.padded and nspread!=0:
            m = self._get_linear_parameters(ntokens-(2*nspread))
        else:
            m = self._get_linear_parameters(ntokens)
        hit_tokens = (hit_times) / m
        hit_tokens = np.round(hit_tokens)
        hit_tokens = hit_tokens.astype(int)
        
        hit_slices = [np.arange(token-nspread, token+nspread+1, dtype=int) for token in hit_tokens]
        hit_slices = np.asarray(hit_slices)
            
        if self.padded:
            hit_slices += nspread
        
        # Handle edge cases
        # In the best case this block should be only executed if padded=False, but now used to prevent errors
        hit_slices[np.where(hit_slices>=ntokens)] = ntokens-1
        hit_slices[np.where(hit_slices<=0)] = 0
        
        return hit_slices
    
    
    def _slicify_event(self, event, ntokens, nspread, non_active=False):
        '''
        Takes an event and puts in into slices
        '''
        n_slices = 2 * nspread + 1
        n_pulses, n_features = event.shape
        hit_slices = self._get_hit_slices(event, ntokens, nspread)
        
        if non_active:
            event_sliced = self.pads * np.ones((ntokens, self.ndoms, n_features-1))
        else:
            event_sliced = self.pads * np.ones((ntokens, n_pulses, n_features))
        
        for i, (pulse, slices) in enumerate(zip(event, hit_slices)):
            if non_active:
                idom = int(pulse[0])
                features = pulse[1:]
            else:
                features = pulse
            
            feat_repeat = np.asarray([features]*n_slices)
                
            if non_active:
                idx = idom
            else:
                idx = i
            event_sliced[slices, idx] = feat_repeat
            
        return event_sliced
    
    
    
    def slicify_events(self, ntokens, nspread, padded=False, non_active=False):
        '''
        Takes pulse list and transforms into temporal slices.
        
        Parameters:
        ----------
        ntokens: int
            Number of tokens to quantify time information in
        nspread: int
            Number of temporal slices a pulse should occupy before and after its actaul slice
        padded: bool
            Reserve pads at beginning and end of hit sequence for possible hit bleeding
        non_active: book
            Include non-active doms (otherwise just pulses)
        '''
        
        self.padded = padded
        events = self.x
        return [self._slicify_event(event, ntokens, nspread, non_active) for event in tqdm(events)]
