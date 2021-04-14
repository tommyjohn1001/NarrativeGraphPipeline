from torch_geometric.nn.glob.glob import global_mean_pool
import torch_geometric.nn as torch_g_nn
import torch.nn as torch_nn
import torch

from configs import args

class GraphReasoning(torch_nn.Module):
    def __init__(self):
        super().__init__()

        self.linear = torch_nn.Linear(768, args.graph_d_project)

        # GCN
        self.gcn1           = torch_g_nn.GCNConv(args.graph_d_project, args.graph_d_project)
        self.gcn2           = torch_g_nn.GCNConv(args.graph_d_project, args.graph_d_project//2)
        self.gcn3           = torch_g_nn.GCNConv(args.graph_d_project//2, args.graph_d_project//2)
        self.gcn4           = torch_g_nn.GCNConv(args.graph_d_project//2, args.graph_d_project//4)
        self.act_leakRelu   = torch_nn.LeakyReLU(inplace=True)

        self.fc     = torch_nn.Sequential(
            torch_nn.Linear(args.graph_d_project//4, args.graph_d_project//4),
            torch_nn.Dropout(args.dropout),
            torch_nn.Linear(args.graph_d_project//4, args.d_hid),
            torch_nn.LeakyReLU()
        )

    def forward(self, node_feat, edge_indx, node_len, edge_len):
        # node_feat : [b, n_nodes, 768]
        # edge_indx : [b, 2, *]
        # edge_len  : [b]
        # node_len  : [b]

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

        # edge_indx   = remove_padded(edge_indx, edge_indx_len)

        node_feat   = self.linear(node_feat)

        node_feat, edge_indx, batch_indx   = self.batchify(node_feat, node_len,
                                                           edge_indx, edge_len)
        X           = gcn(node_feat, edge_indx)
        mean_pool   = global_mean_pool(X, batch_indx)
        # [b, graph_d_project//4]

        final_vec   = self.fc(mean_pool)
        # [b, d_hid]

        return final_vec


    def batchify(self, node_feat:torch.Tensor, node_len:torch.Tensor, edge_indx:torch.Tensor, edge_len:torch.Tensor) -> tuple:
        """Convert batch of node features and edge indices into a big graph"""

        # node_feat : [b, n_nodes, d]
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
