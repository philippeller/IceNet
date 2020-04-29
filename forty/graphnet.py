import torch.nn as NN
import torch_geometric.nn as geo_nn

class GraphNet(torch.nn.Module):
    def __init__(self, config):
        super(GraphNet, self).__init__()
        self.config = config
        self.activations = {'relu': NN.ReLU, 'leaky': NN.LeakyReLU}
        self.pools = {'max': geo_nn.global_max_pool, 'mean': geo_nn.global_mean_pool, 'add': geo_nn.global_add_pool}
        self.graph_layers, self.fc_layers = self._setup_layers()

    def _setup_layers(self):
        dims = []
        # GNN layers
        graph_layers = []
        for layer in self.config['graph_layers']:
            if type(layer) == int:
                dim = dims[-1] if len(dims)!=0 else self.config['n_features']
                graph_layers.append(geo_nn.SAGEConv(dim, layer))
                dims.append(layer)

            elif type(layer) == str and layer in self.activations:
                graph_layers.append(self.activations[layer]())

        # Fully connected layers
        fc_layers = []
        for i, layer in enumerate(self.config['fc_layers']):
            if type(layer) == int:
                if type(self.config['fc_layers'][i-1]) == int and i!=0:
                    fc_layers.append(NN.ReLU())
                    continue

                dim = dims[-1] if len(dims)!=0 else self.config['n_features']
                fc_layers.append(NN.Linear(dim, layer))
                dims.append(layer)

            elif type(layer) == str and layer in self.activations:
                fc_layers.append(self.activations[layer]())

            elif layer == 'drop':
                fc_layers.append(NN.Dropout(.3))

        # Output layer
        fc_layers.append(NN.Linear(dims[-1], config['n_labels']))

        return NN.ModuleList(graph_layers), NN.ModuleList(fc_layers)


    def forward(self, data):
        x, edge_index, edge_weights = data.x, data.edge_index, data.edge_attr
        batch = data.batch

        for layer in self.graph_layers:
            if isinstance(layer, SAGEConv):
                x = layer(x, edge_index, edge_weights)
            else:
                x = layer(x)
        x = self.pools[self.config['pooling']](x, batch)
        for layer in self.fc_layers:
            x = layer(x)

        x = x.view(-1)

        return x
