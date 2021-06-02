from typing import Any, Optional
import copy


from torch.nn.modules.transformer import Tensor, F, MultiheadAttention,\
    ModuleList, Dropout, Linear, LayerNorm
from torch.nn.parameter import Parameter
import torch.nn as torch_nn
import torch


class CustomTransEnc(torch_nn.Module):
    r"""CustomTransEnc is a stack of N enc layers

    Args:
        enc_layer: an instance of the CustomTransEncLayer() class (required).
        num_layers: the number of sub-enc-layers in the enc (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> enc_layer = nn.CustomTransEncLayer(d_model=512, nhead=8)
        >>> trans_enc = nn.CustomTransEnc(enc_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = trans_enc(src)
    """
    __constants__ = ['norm']

    def __init__(self,
        enc_layer,
        num_layers,
        seq_len_ques: int = 42,
        seq_len_para: int = 122,
        d_hid: int = 64,
        norm=None):
        super(CustomTransEnc, self).__init__()
        self.layers     = _get_clones(enc_layer, num_layers)
        self.num_layers = num_layers
        self.norm       = norm
        self.fc         = torch_nn.Sequential(
            torch_nn.Linear(d_hid, d_hid),
            torch_nn.ReLU(),
            torch_nn.BatchNorm1d(seq_len_ques + seq_len_para)
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

        output1 = output1.transpose(0, 1)
        # [b, seq_len_ques + seq_len_para, d_hid]
        output1 = self.fc(output1)

        return output1

def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

class CustomTransEncLayer(torch_nn.Module):
    r"""CustomTransEncLayer is made up of self-attn and feedforward network.
    This standard enc layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> enc_layer = nn.CustomTransEncLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = enc_layer(src)
    """

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

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(CustomTransEncLayer, self).__setstate__(state)

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




class Memory(torch_nn.Module):
    def __init__(self,
        seq_len_ques: int = 42,
        seq_len_para: int = 122,
        n_layers_gru: int = 5,
        d_hid: int = 64,
        n_paras: int = 30,
        device: Any = None):
        super().__init__()

        self.d_hid          = d_hid
        self.seq_len_ques   = seq_len_ques
        self.seq_len_para   = seq_len_para

        self.memory     = Parameter(torch.zeros((n_paras, seq_len_ques + seq_len_para, d_hid), device=device), requires_grad=False)

        self.linear1    = torch_nn.Linear(d_hid, d_hid, bias=False)
        self.linear2    = torch_nn.Linear(d_hid, d_hid, bias=False)
        self.linear_gate= torch_nn.Linear(d_hid, d_hid, bias=True)
        self.biGRU_mask = torch_nn.GRU(d_hid, d_hid//2, num_layers=n_layers_gru,
                                       batch_first=True, bidirectional=True)

    def forward(self):
        return

    def update_mem(self, nth_para: int, ques_para: Tensor, output: Tensor):
        """Update memory

        Args:
            nth_para (int): no. dim to be updated in list 'memory'
            ques_para (Tensor): combination of quesiton and para
            output (Tensor): output of TransEnc
        """
        # ques_para : [b, seq_len_ques + seq_len_para, d_hid]
        # output    : [b, seq_len_ques + seq_len_para, d_hid]

        ques_para   = torch.sum(ques_para, dim=0)
        # [seq_len_ques + seq_len_para, d_hid]
        output      = torch.sum(output, dim=0)
        # [seq_len_ques + seq_len_para, d_hid]

        gate    = self.linear1(ques_para) + self.linear2(output)
        gate    = torch.sigmoid(self.linear_gate(gate))

        tmp = gate * ques_para + (1 - gate) * self.memory[nth_para]
        self.memory[nth_para] = tmp.detach()


    def get_final_memory(self, ques, paras, paras_mask):
        # ques      : [b, seq_len_ques, d_hid]
        # paras     : [b, n_paras, seq_len_para, d_hid]
        # paras_mask: [b, n_paras, seq_len_para]

        b, n_paras, seq_len_para, _ = paras.shape

        #########################
        # Calculate CoAttention matrix w.r.t. ques and paras
        # to infer attentive mask
        #########################
        paras_      = paras.reshape(-1, seq_len_para, self.d_hid)
        paras_len   = torch.sum(paras_mask, dim=2).reshape(-1).to('cpu')
        # paras_    : [b*n_paras, seq_len_para, d_hid]
        # paras_mask: [b*n_paras]

        # for i in range(paras_len.shape[0]):
        #     if paras_len[i] == 0:
        #         paras_len[i] = 1

        # tmp     = torch_nn.utils.rnn.pack_padded_sequence(paras_, paras_len, batch_first=True,
        #                                                   enforce_sorted=False)
        # tmp     = self.biGRU_mask(tmp)[0]
        # paras_  = torch_nn.utils.rnn.pad_packed_sequence(tmp, batch_first=True)[0]
        paras_    = self.biGRU_mask(paras_)[0]
        # [b*n_paras, seq_len_para, d_hid]

        paras_first = paras_[:, 0, :].reshape(b, n_paras, -1)
        # [b, n_paras, d_hid]

        ## Calculate Affinity matrix
        A   = torch.bmm(ques, paras_first.transpose(1, 2)).softmax(dim=1)
        # [b, seq_len_ques, n_paras]

        #########################
        # Infer attentive mask from Affinity matrix
        #########################
        attentive_mask = torch.bmm(A.transpose(1, 2), ques)
        # [b, n_paras, d_hid]

        ## Apply some transformations to mask
        attentive_mask = attentive_mask.sum(dim=0).softmax(0)
        # [n_paras, d_hid]
        
        attentive_mask = attentive_mask.unsqueeze(1).repeat(1, self.seq_len_ques + seq_len_para, 1)
        # [n_paras, seq_len_ques + seq_len_para, d_hid]


        #########################
        # Apply attentive mask
        #########################
        final_memory   = (self.memory * attentive_mask).sum(dim=0)
        # [seq_len_ques + seq_len_para, d_hid]

        
        return final_memory

    def get_mem_cell(self, nth_para):
        return self.memory[nth_para]
