from itertools import combinations
from typing import Any

import torch.nn as torch_nn
import torch

from src.models.layers.reasoning_layer.sub_layers import GraphLayer, Memory


class GraphBasedMemoryLayer(torch_nn.Module):
    def __init__(self,
        batch_size: int = 5,
        seq_len_ques: int = 42,
        seq_len_ans: int = 42,
        d_hid: int = 64,
        d_bert: int = 768,
        d_graph: int = 2048,
        n_nodes: int = 435,
        n_edges: int = 3120,
        device: Any = None
    ):
        super().__init__()

        self.n_nodes    = n_nodes
        self.d_hid      = n_nodes
        self.d_bert     = d_bert

        self.lin1       = torch_nn.Linear(d_bert*2, d_hid, bias=False)

        self.graph      = GraphLayer(d_hid, d_graph, device)
        self.memory     = Memory(batch_size, n_nodes, d_hid, n_edges, device)

        self.lin2       = torch_nn.Linear(d_bert, seq_len_ans, bias=False)
        self.lin3       = torch_nn.Linear(seq_len_ques, n_nodes, bias=False)
        self.lin4       = torch_nn.Linear(d_hid, d_bert, bias=False)


    def forward(self, ques, paras):
        # ques : [b, seq_len_ques, d_bert]
        # paras: [b, n_paras, d_bert]

        b, n_paras, _, = paras.shape


        ######################################
        # Use TransformerEncoder to encode
        # question and paras
        ######################################
        ques_   = torch.mean(ques, dim=1).unsqueeze(1).repeat(1, n_paras, 1)
        X       = torch.cat((paras, ques_), dim=2)
        # [b, n_paras, d_bert*2]
        X       = self.lin1(X)
        # [b, n_paras, d_hid]


        ######################################
        # Get node feat and edge indx from memory
        # and combine with X to create tensors for Graph
        ######################################
        # Get things from memory
        node_feats_mem, edge_indx, edge_len = self.memory.gets()
        # node_feats_mem    : [batch, n_nodes, d_hid]
        # edge_indx         : [batch, 2, n_edges]
        # edge_len          : [batch]

        node_feats_mem  = node_feats_mem[:b, :, :]
        edge_indx       = edge_indx[:b, :, :]
        edge_len        = edge_len[:b]
        # node_feats_mem    : [b, n_nodes, d_hid]
        # edge_indx         : [b, 2, n_edges]
        # edge_len          : [b]


        # Create node feat from tensor X
        node_feats   = []

        for pair in combinations(range(n_paras), 2):
            idx1, idx2 = pair
            node_feats.append(torch.cat([X[:, idx1, :], X[:, idx2, :]], dim=1).unsqueeze(1))


        node_feats  = torch.cat(node_feats, dim=1)
        # [b, n_nodes, d_hid*2]
        node_len    = torch.IntTensor([self.n_nodes]).repeat(b)
        # [b]

        # Concat 'node_feats' with 'node_feats_mem'
        node_feats  = torch.cat((node_feats, node_feats_mem), dim=2)
        # [b, n_nodes, d_hid*3]


        ######################################
        # Pass through Graph
        ######################################
        Y   = self.graph(node_feats, edge_indx, node_len, edge_len)
        # [b, n_nodes, d_hid]


        ######################################
        # Update memory
        ######################################
        self.memory.update_mem(Y)


        ######################################
        # Derive attentive matrix from question
        # for tensor 'Y'
        ######################################
        ques    = self.lin2(ques)
        # [b, seq_len_ques, seq_len_ans]
        attentive = self.lin3(ques.transpose(1, 2))
        # [b, seq_len_ans, n_nodes]

        Y       = torch.bmm(torch.softmax(attentive, dim=2), Y)
        # [b, seq_len_ans, d_hid]

        Y       = self.lin4(Y)
        # [b, seq_len_ans, d_bert]


        return Y

    def save_memory(self):
        self.memory.save_memory()
