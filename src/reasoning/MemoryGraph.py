from itertools import combinations

import torch.nn as torch_nn
import torch

from src.reasoning.layers import Graphlayer, TransEncLayer, Memory
from configs import args



class GraphBasedMemoryReasoning(torch_nn.Module):
    def __init__(self):
        super().__init__()

        self.n_nodes    = args.n_nodes


        self.trans_enc  = TransEncLayer()
        self.graph      = Graphlayer()
        self.memory     = Memory()


    def forward(self, ques, paras):
        # ques  : [b, seq_len_ques, d_hid]
        # paras : [b, n_paras, seq_len_para, d_hid]

        b, n_paras, _, d_hid    = paras.shape


        ######################################
        # Use TransformerEncoder to encode
        # question and paras
        ######################################
        X   = self.trans_enc(ques, paras).transpose(0, 1)
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
        # Concate tensor 'Y' with 'paras;
        ######################################
        paras   = paras.view(b, -1, d_hid)
        # [b, n_paras*seq_len_para, d_hid]

        Y       = torch.cat((Y, paras), dim=1)
        # [b, n_nodes + n_paras*seq_len_para, d_hid]



        return Y

    def save_memory(self):
        self.memory.save_memory()
