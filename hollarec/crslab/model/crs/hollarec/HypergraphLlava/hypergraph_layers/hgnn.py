import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import degree
from torch_geometric.data import Data

from transformers.configuration_utils import PretrainedConfig

def hgnn_conv(x, hyperedge_index, hyperedge_weight=None, num_nodes=None):
    """
    Args:
        x (torch.Tensor): 
            Node feature matrix
            shape (num_nodes, in_channels).
        hyperedge_index (torch.Tensor): 
            The hypergraph connectivity in COO format
            shape (2, num_hyperedges).
        hyperedge_weight (torch.Tensor):
            The weight of each hyperedge,
            shape (num_hyperedges,).
        num_nodes (int):
            The number of nodes in the graph.

    Returns:
        torch.Tensor: The updated node feature matrix, shape (num_nodes, out_channels).
    """
    
    if num_nodes is None:
        num_nodes = x.size(0)
    
    node_idx, hyperedge_idx = hyperedge_index
    num_hyperedges = hyperedge_idx.max().item() + 1

    d_v = degree(index=node_idx, num_nodes=num_nodes)
    d_v_inv_sqrt = d_v.pow(-0.5)
    d_v_inv_sqrt[torch.isinf(d_v_inv_sqrt)] = 0

    d_e = degree(index=hyperedge_idx, num_nodes=num_hyperedges)
    d_e_inv = d_e.pow(-1.0)
    d_e_inv[torch.isinf(d_e_inv)] = 0

    x_transformed = d_v_inv_sqrt.view(-1, 1) * x

    # Aggregate node features to hyperedges
    hyperedge_features = torch.zeros((num_hyperedges, x.size(1)), device=x.device)
    hyperedge_features = hyperedge_features.index_add(0, hyperedge_idx, x_transformed[node_idx])

    if hyperedge_weight is not None:
        hyperedge_features = hyperedge_weight.view(-1, 1) * hyperedge_features

    hyperedge_features_transformed = d_e_inv.view(-1, 1) * hyperedge_features

    # Aggregate hyperedge features back to nodes
    x_out = torch.zeros((num_nodes, x.size(1)), device=x.device)
    x_out = x_out.index_add(0, node_idx, hyperedge_features_transformed[hyperedge_idx])

    # Final normalization
    x_out = d_v_inv_sqrt.view(-1, 1) * x_out

    return x_out

class HGNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_bias: bool = True,
    ):
        super(HGNN, self).__init__()
        self.config = PretrainedConfig()
        self.num_layers = num_layers
        self.dropout = dropout
        self.activation = nn.ReLU()
        
        self.build_layers(in_channels, hidden_channels, out_channels, num_layers, use_bias)
        self.reset_parameters()

    def build_layers(self, in_channels, hidden_channels, out_channels, num_layers, use_bias):
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_channels, hidden_channels, bias=use_bias))
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_channels, hidden_channels, bias=use_bias))
        self.layers.append(nn.Linear(hidden_channels, out_channels, bias=use_bias))

    def reset_parameters(self):
        for layer in self.layers:
            # layer.reset_parameters()
            nn.init.xavier_uniform_(layer.weight, gain=1.414)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, data: Data):
        x = data.x
        hyperedge_index = data.hyperedge_index

        hyperedge_weight = getattr(data, 'hyperedge_weight', None)

        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = hgnn_conv(x, hyperedge_index, hyperedge_weight)
            if i != self.num_layers - 1:
                x = self.activation(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

