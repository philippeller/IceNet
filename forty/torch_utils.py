import numpy as np


def torch_to_numpy(x):
    return np.asarray(x.cpu().detach())
