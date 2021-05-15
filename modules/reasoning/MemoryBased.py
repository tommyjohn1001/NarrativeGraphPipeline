import copy
from typing import Optional


from torch.nn.modules.transformer import Tensor, F, MultiheadAttention,\
    ModuleList, Dropout, Linear, LayerNorm
import torch.nn as torch_nn
import torch

from modules.utils import check_exist
from configs import args, PATH, logging


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

    def __init__(self, enc_layer, num_layers, norm=None):
        super(CustomTransEnc, self).__init__()
        self.layers     = _get_clones(enc_layer, num_layers)
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

        return output1


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


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


class Memory(torch_nn.Module):
    def __init__(self):
        super().__init__()

        self.memory     = [None]*(args.n_paras)
        self.attn_mask  = torch.rand((args.n_paras, 1, args.batch, 1),
                                     device=args.device, requires_grad=True)

        if check_exist(PATH['memory']):
            logging.info("Tensors of MemoryModule exist. Load them.")

            backup  = torch.load(PATH['memory'], map_location="cuda:0")
            self.memory     = backup['memory']
            self.attn_mask  = backup['attn_memory']


        self.linear1    = torch_nn.Linear(args.d_hid, args.d_hid, bias=False)
        self.linear2    = torch_nn.Linear(args.d_hid, args.d_hid, bias=False)
        self.linear_gate= torch_nn.Linear(args.d_hid, args.d_hid, bias=True)


    def forward(self, nth_para: int, ques_para: Tensor, output: Tensor):
        # print(f"output   : {output.shape}")
        # print(f"ques_para: {ques_para.shape}")
        gate    = self.linear1(ques_para) + self.linear2(output)
        gate    = torch.sigmoid(self.linear_gate(gate))

        if  self.memory[nth_para] is None:
            self.memory[nth_para] = ques_para.detach()
        else:
            tmp = gate * ques_para + (1 - gate) * self.memory[nth_para]
            print()
            self.memory[nth_para] = tmp.detach()


    def update_mem(self, nth_para: int, ques_para: Tensor, output: Tensor):
        """Update memory

        Args:
            nth_para (int): no. dim to be updated in list 'memory'
            ques_para (Tensor): combination of quesiton and para
            output (Tensor): output of TransEnc
        """

        self.forward(nth_para, ques_para, output)

    def get_final_memory(self):
        attn_mask   = torch.softmax(self.attn_mask, dim=0)
        attn_mask   = attn_mask.repeat(1, args.seq_len_ques + args.seq_len_para, 1, args.d_hid)
        # [n_paras, seq_len_ques + seq_len_para, b, d_hid]

        memory_     = [t.unsqueeze(0) for t in self.memory]
        memory_     = torch.vstack(memory_)
        # [n_paras, seq_len_ques + seq_len_para, b, d_hid]

        return (memory_ * attn_mask).sum(dim=0)

    def get_mem_cell(self, nth_para):
        return self.memory[nth_para]

    def save_memory(self):
        torch.save({
            'memory'        : self.memory,
            'attn_memory'   : self.attn_mask
        }, PATH['memory'])


class MemoryBasedReasoning(torch_nn.Module):
    def __init__(self):
        super().__init__()

        self.mlp_ques_para  = torch_nn.Sequential(
            torch_nn.Linear(args.d_hid, args.d_hid),
            torch_nn.GELU(),
            torch_nn.Dropout(args.dropout)
        )

        self.trans_enc      = CustomTransEnc(CustomTransEncLayer(d_model=args.d_hid, nhead=args.trans_nheads),
                                             num_layers=args.trans_nlayers)


        self.memory     = Memory()

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

        Y = []

        for nth_para in range(args.n_paras):
            ques_para   = ques_paras[nth_para]
            # [seq_len_ques + seq_len_para, b, d_hid]
            if self.memory.get_mem_cell(nth_para) is None:
                memory_cell = ques_para.squeeze(0)
            else:
                memory_cell = self.memory.get_mem_cell(nth_para).squeeze(0)
            # [seq_len_ques + seq_len_para, b, d_hid]

            output      = self.trans_enc(memory_cell, ques_para)
            # [seq_len_ques + seq_len_para, b, d_hid]

            self.memory.update_mem(nth_para, ques_para, output)

            Y.append(output.unsqueeze(0))

        final   = self.memory.get_final_memory()
        # [seq_len_ques + seq_len_para, args.batch, d_hid]

        # We do not take entire tensor 'final' but chunk of it
        # because tensor 'final' is for batch_size whereas in last batch step,
        # the number of datapoints are not equal batch_size
        Y.append(final[:, :b, :].unsqueeze(0))
        Y   = torch.cat(Y, dim=0).to(args.device).permute(0, 2, 1, 3)
        # [n_paras + 1, b, seq_len_ques + seq_len_para, d_hid]
        print(f"Y: {Y.shape}")
        Y   = Y.reshape(n_paras + 1, b, -1)
        # [n_paras+1, b, (seq_len_ques + seq_len_para)*d_hid]

        Y   = self.linearY(Y)
        # [n_paras + 1, b, d_hid]


        print(f"Y = {Y.shape}")

        return Y

    def save_memory(self):
        self.memory.save_memory()
