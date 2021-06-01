"""
Graph Attention Networks in DGL library.
References
----------
GAT implementation of DGL based on tutorial provided below
https://docs.dgl.ai/en/0.4.x/tutorials/models/1_gnn/9_gat.html
https://github.com/dmlc/dgl/blob/master/examples/pytorch/gat/gat.py
"""

import torch.nn as nn
from dgl.nn.pytorch import GATConv


class GAT(nn.Module):
    def __init__(self, graph, num_layers, input_dimensions, hidden_units, num_labels, attention_heads, activation,
                 input_drop, attention_drop_rate, alpha, residual):
        super(GAT, self).__init__()
        self.g = graph
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        self.gat_layers.append(GATConv(input_dimensions, hidden_units, attention_heads[0], input_drop,
                                       attention_drop_rate, alpha, False, self.activation))
        """Build the hidden graph attention layers of the network"""
        for layer in range(1, num_layers):
            self.gat_layers.append(GATConv(hidden_units * attention_heads[layer-1], hidden_units, attention_heads[layer],
                                           input_drop, attention_drop_rate, alpha, residual, self.activation))
        """Aggregating the attention at the output"""
        self.gat_layers.append(GATConv(hidden_units * attention_heads[-2], num_labels, attention_heads[-1], input_drop,
                                       attention_drop_rate, alpha, residual, None))

    def forward(self, inputs):
        h = inputs
        for layer in range(self.num_layers):
            h = self.gat_layers[layer](self.g, h).flatten(1)
        """Extracting the Aggregate attention from the final layer"""
        logits = self.gat_layers[-1](self.g, h).mean(1)
        return logits

