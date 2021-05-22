

from typing import Optional
from itertools import combinations
from random import sample
import copy

from torch.nn.modules.transformer import Tensor, F, MultiheadAttention,\
    ModuleList, Dropout, Linear, LayerNorm
import torch_geometric.nn as torch_g_nn
import torch.nn as torch_nn
import torch
import numpy as np

from src.utils import check_exist
from configs import args, PATH, logging



class Graphlayer(torch_nn.Module):
    def __init__(self):
        super().__init__()

        self.linear = torch_nn.Linear(args.d_hid*3, args.d_graph)

        # GCN
        self.gcn1           = torch_g_nn.GCNConv(args.d_graph, args.d_graph)
        self.gcn2           = torch_g_nn.GCNConv(args.d_graph, args.d_graph//2)
        self.gcn3           = torch_g_nn.GCNConv(args.d_graph//2, args.d_graph//2)
        self.gcn4           = torch_g_nn.GCNConv(args.d_graph//2, args.d_graph//4)
        self.act_leakRelu   = torch_nn.LeakyReLU(inplace=True)

        self.linear2        = torch_nn.Linear(args.d_graph//4, args.d_hid, bias=False)

    def forward(self, node_feat, edge_indx, node_len, edge_len):
        # node_feat : [b, n_nodes, d_hid*3]
        # edge_indx : [b, 2, n_edges]
        # node_len  : [b]
        # edge_len  : [b]

        b, n_nodes, d_hid   = node_feat.shape
        d_hid = d_hid // 3

        def gcn(node_feats, edge_indx):
            X   = self.gcn1(node_feats, edge_indx)
            X   = self.act_leakRelu(X)

            X   = self.gcn2(X, edge_indx)
            X   = self.act_leakRelu(X)

            X   = self.gcn3(X, edge_indx)
            X   = self.act_leakRelu(X)

            X   = self.gcn4(X, edge_indx)
            X   = self.act_leakRelu(X)

            return X

        node_feat   = self.linear(node_feat)

        node_feat, edge_indx, _ = self.batchify(node_feat, node_len,
                                                edge_indx, edge_len)

        X   = gcn(node_feat, edge_indx)
        # [n_nodes * b, d_graph//4]

        X   = self.linear2(X).view(b, n_nodes, d_hid)
        # [b, n_nodes, d_hid]

        return X


    def batchify(self, node_feat:torch.Tensor, node_len:torch.Tensor, edge_indx:torch.Tensor, edge_len:torch.Tensor) -> tuple:
        """Convert batch of node features and edge indices into a big graph"""
        # node_feat : [b, n_nodes, d_hid*3]
        # node_len  : [b]
        # edge_indx : [b, 2, *]
        # edge_len  : [b]
        batch = node_feat.shape[0]

        accum = 0
        final_edge_indx = None
        final_node_feat = None
        batch_indx      = []
        for b in range(batch):
            ## 1. accummulate node feature
            ## 1.1. get node feat of that batch and remove padding
            node_feat_  = node_feat[b, :node_len[b].item(), :].squeeze(0)

            ## 1.3. Concate into 'final_node_feat'
            if final_node_feat is None:
                final_node_feat = node_feat_
            else:
                final_node_feat = torch.vstack((final_node_feat, node_feat_))


            ## 2. accummulate edge indx
            ## 2.1. get edge indx of that batch and remove padding
            edge_indx_  = edge_indx[b, :, :edge_len[b].item()].squeeze(0)

            ## 2.2. Increment index of that edge indx by accum
            increment   = torch.Tensor([accum]).repeat(edge_indx_.shape).to(args.device)
            edge_indx_  = edge_indx_ + increment

            ## 2.3. Concate into 'final_edge_indx'
            if final_edge_indx is None:
                final_edge_indx = edge_indx_
            else:
                final_edge_indx = torch.hstack((final_edge_indx, edge_indx_))

            ## 3. Update batch_indx and accum
            batch_indx = batch_indx + [b]*(node_len[b].item())
            accum += node_len[b].item()

        return  final_node_feat.to(args.device), final_edge_indx.to(args.device).long(),\
                torch.LongTensor(batch_indx).to(args.device)



class CustomTransEnc(torch_nn.Module):
    __constants__ = ['norm']

    def __init__(self, enc_layer, num_layers, norm=None):
        super(CustomTransEnc, self).__init__()
        self.layers     = self._get_clones(enc_layer, num_layers)
        self.num_layers = num_layers
        self.norm       = norm
        self.fc         = torch_nn.Sequential(
            torch_nn.Linear(args.d_hid, args.d_hid),
            torch_nn.ReLU(),
            torch_nn.Dropout(args.dropout)
        )

    def forward(
        self,
        query: Tensor,
        key_value: Tensor,
        query_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the enc layers in turn.

        Args:
            query: the query sequence to the enc (required).
            key_value: the key/value sequence to the enc (required).
            query_mask: the mask for the query sequence (optional).
            src_key_padding_mask: the mask for the query keys per batch (optional).

        Shape:
            see the docs in Trans class.
        """
        output1, output2 = query, key_value

        for mod in self.layers:
            output1, output2 = mod(output1, output2, query_mask=query_mask,
                                   src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output1 = self.norm(output1)

        output1 = self.fc(output1)
        # [seq_len_ques + seq_len_para, batch, d_hid]

        return output1

    def _get_clones(self, module, N):
        return ModuleList([copy.deepcopy(module) for i in range(N)])

class CustomTransEncLayer(torch_nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(CustomTransEncLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = self._get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(CustomTransEncLayer, self).__setstate__(state)

    def _get_activation_fn(self, activation):
        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu

        raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

    def forward(self,
        query: Tensor,
        key_value: Tensor,
        query_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the enc layer.

        Args:
            src: the sequence to the enc layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Trans class.
        """
        src2 = self.self_attn(query, key_value, key_value, attn_mask=query_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        query = query + self.dropout1(src2)
        query = self.norm1(query)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
        query = query + self.dropout2(src2)
        query = self.norm2(query)
        return query, query

class TransEncLayer(torch_nn.Module):
    def __init__(self):
        super().__init__()

        self.trans_enc  = CustomTransEnc(CustomTransEncLayer(d_model=args.d_hid, nhead=args.trans_nheads),
                                         num_layers=args.trans_nlayers)

        self.linearY    = torch_nn.Sequential(
            torch_nn.Linear((args.seq_len_ques + args.seq_len_para) * args.d_hid,\
                            args.d_hid),
            torch_nn.GELU(),
            torch_nn.Dropout(args.dropout)
        )

    def forward(self, ques, paras):
        # ques  : [b, seq_len_ques, d_hid]
        # paras : [b, n_paras, seq_len_para, d_hid]

        b, n_paras, _, _ = paras.shape


        ## Unsqueeze and repeat tensor 'ques' to match shape of 'paras'
        ques         = ques.unsqueeze(1).repeat(1, args.n_paras, 1, 1)
        ques_paras   = torch.cat((ques, paras), dim=2).permute(1, 2, 0, 3)
        # [n_paras, seq_len_ques + seq_len_para, b, d_hid]


        X = []

        for nth_para in range(args.n_paras):
            ques_para   = ques_paras[nth_para]
            # [seq_len_ques + seq_len_para, batch, d_hid]

            output      = self.trans_enc(ques_para, ques_para)
            # [seq_len_ques + seq_len_para, batch, d_hid]

            X.append(output.unsqueeze(0))


        X   = torch.cat(X, dim=0).to(args.device).permute(0, 2, 1, 3)
        # [n_paras, b, seq_len_ques + seq_len_para, d_hid]

        X   = X.reshape(n_paras, b, -1)
        # [n_paras, b, (seq_len_ques + seq_len_para)*d_hid]

        X   = self.linearY(X)
        # [n_paras, b, d_hid]

        return X



class Memory(torch_nn.Module):
    def __init__(self):
        super().__init__()

        self.batch      = args.batch
        self.n_nodes    = args.n_nodes
        self.d_hid      = args.d_hid
        self.n_edges    = args.n_edges

        self.edge_len   = torch.IntTensor([args.n_edges]).repeat(self.batch).to(args.device)

        if check_exist(PATH['memory']):
            logging.info("Tensors of MemoryModule exist. Load them.")

            tmp = torch.load(PATH['memory'])
            self.node_feats_mem = tmp['node_feats_mem'].to(args.device)
            self.edge_index     = tmp['edge_index'].to(args.device)
        else:
            self.node_feats_mem = torch.rand(self.batch, self.n_nodes, self.d_hid).to(args.device)
            self.edge_index     = self.gen_edges().to(args.device)


    def forward(self):
        pass

    def update_mem(self, Y):
        """Update memory with given tensor

        Args:
            Y (Tensor): output of Graph module
        """
        # Y: [b, n_nodes, d_hid]
        b   = Y.shape[0]
        if b < args.batch:
            tmp = self.node_feats_mem[b:, :, :]
            Y   = torch.cat((Y, tmp), dim=0)
        # Y: [batch, n_nodes, d_hid]

        self.node_feats_mem = Y.detach()

    def save_memory(self):
        torch.save({
            'node_feats_mem'    : self.node_feats_mem,
            'edge_index'        : self.edge_index
        }, PATH['memory'])

    def gen_edges(self):
        edge_pair   = list(combinations(range(435), 2))
        edges       = sample(edge_pair, self.n_edges//2)

        vertex_s, vertex_d = [], []
        for edge in edges:
            s, d = edge
            vertex_s.append(int(s))
            vertex_d.append(int(d))

            vertex_s.append(int(d))
            vertex_d.append(int(s))

        edge_index  = np.array([vertex_s, vertex_d])
        # [2, *]

        edge_index  = torch.from_numpy(edge_index).unsqueeze(0).repeat(self.batch, 1, 1)

        return edge_index

    def gets(self):
        return self.node_feats_mem, self.edge_index, self.edge_len
