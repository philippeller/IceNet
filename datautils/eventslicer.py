class Tokenizer:
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
        self.tmin = np.min(min_times[min_times!=0])
        self.tmax = np.max(max_times[max_times!=0])
        self.ndoms = 5160
        self.pads = pads
    
    
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
        t = self.tmin
        m = (self.tmax - self.tmin) / ntokens
        return m, t
    
    
    def _get_hit_slices(self, event, ntokens, nspread, padded=False):
        hit_times = event[:,1]
        # If padded, reserve spots for spread values
        if padded:
            m, t = self._get_linear_parameters(ntokens-2*nspread-1)
        else:
            m, t = self._get_linear_parameters(ntokens)
        hit_tokens = (hit_times - t) / m
        hit_tokens = np.round(hit_tokens)
        hit_tokens = hit_tokens.astype(int)
        
        hit_slices = [np.arange(token-nspread, token+nspread+1, dtype=int) for token in hit_tokens]
        hit_slices = np.asarray(hit_slices)
            
        if padded:
            hit_slices += nspread
        
        # Handle edge cases
        # In the best case this block should be only executed if padded=False, but now used to prevent errors
        hit_slices[np.where(hit_slices>=ntokens)] = ntokens-1
        hit_slices[np.where(hit_slices<=0)] = 0
        
        return hit_slices
    
    
    def _slicify_event(self, event, ntokens, nspread, padded=False):
        n_slices = 2 * nspread + 1
        n_pulses, n_features = event.shape
        hit_slices = self._get_hit_slices(event, ntokens, nspread)
        
        event_sliced = self.pads * np.ones((ntokens, self.ndoms, n_features-1))
        for pulse, slices in zip(event, hit_slices):
            idom = int(pulse[0])
            features = pulse[1:]
            feat_repeat = np.repeat(features, n_slices).reshape(n_slices,-1)
            event_sliced[slices, idom] = feat_repeat
        
        return event_sliced
    
    
    def slicify_events(self, ntokens, nspread, padded=False):
        events = x
        return [self._slicify_event(event, ntokens, nspread, padded) for event in tqdm(events)]
