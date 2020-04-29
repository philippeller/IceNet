import numpy as np
import data_preparation
from torch_geometric import Data, Dataloader


def get_data_list(features,weight_matrices, y)
    data_list = []
    for features, w, truth in tqdm(zip(features, weight_matrices, y), total=len(y)):
        edge_index, edge_weights = get_edge_info(w)
        dd = Data(x=torch.tensor(features, dtype=torch.float),
                  y=torch.tensor(truth, dtype=torch.float),
                  edge_index=torch.tensor(edge_index, dtype=torch.long),
                  edge_attr=torch.tensor(edge_weights, dtype=torch.float),
                 )
        data_list.append(dd)


def get_k_fold_sets(dataset, k, i, n_samples=None):
    n_samples = n_samples if n_samples else len(dataset)
    n_samples = n_samples - n_samples%k
    dataset = dataset[:n_samples]
    n_subset = n_samples // k
    val_mask = np.arange(i*n_subset, (i+1)*n_subset)
    val_set = [data for i, data in enumerate(dataset) if i in val_mask]
    test_set = [data for i, data in enumerate(dataset) if i not in val_mask]
    return test_set, val_set



