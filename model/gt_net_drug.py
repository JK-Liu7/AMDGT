import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

"""
    Graph Transformer with edge features
    
"""
import dgl
from model.graph_transformer_layer import GraphTransformerLayer


class GraphTransformer(nn.Module):
    def __init__(self, device, n_layers, node_dim, hidden_dim, out_dim, n_heads, dropout):
        super(GraphTransformer, self).__init__()

        self.device = device
        self.layer_norm = True
        self.batch_norm = False
        self.residual = True
        self.linear_h = nn.Linear(node_dim, hidden_dim)
        # self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        self.layers = nn.ModuleList([GraphTransformerLayer(hidden_dim, hidden_dim, n_heads, dropout, self.layer_norm,
                                                           self.batch_norm, self.residual)
                                     for _ in range(n_layers - 1)])
        self.layers.append(
            GraphTransformerLayer(hidden_dim, out_dim, n_heads, dropout, self.layer_norm, self.batch_norm,
                                  self.residual))

    def forward(self, g):
        # input embedding
        g = g.to(self.device)
        h = g.ndata['drs'].float().to(self.device)

        h = self.linear_h(h)
        # h = self.in_feat_dropout(h)

        # convnets
        for conv in self.layers:
            h= conv(g, h)

        g.ndata['h'] = h

        # h = dgl.mean_nodes(g, 'h')

        return h
