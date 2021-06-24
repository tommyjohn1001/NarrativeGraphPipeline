from typing import Any, Optional
import copy


from torch.nn.modules.transformer import (
    Tensor,
    F,
    MultiheadAttention,
    ModuleList,
    Dropout,
    Linear,
    LayerNorm,
)
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
    __constants__ = ["norm"]

    def __init__(
        self,
        enc_layer,
        num_layers,
        d_hid: int = 64,
        norm=None,
    ):
        super(CustomTransEnc, self).__init__()
        self.layers = _get_clones(enc_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.fc = torch_nn.Sequential(
            torch_nn.Linear(d_hid, d_hid),
            torch_nn.ReLU(),
            torch_nn.BatchNorm1d(1),
        )

    def forward(
        self,
        query: Tensor,
        key_value: Tensor,
        query_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Pass the input through the enc layers in turn.

        Args:
            query: the query sequence to the enc (required).
            key_value: the key/value sequence to the enc (required).
            query_mask: the mask for the query sequence (optional).
            src_key_padding_mask: the mask for the query keys per batch (optional).

        Shape:
            see the docs in Trans class.
        """
        output1, output2 = query.transpose(0, 1), key_value.transpose(0, 1)

        for mod in self.layers:
            output1, output2 = mod(
                output1,
                output2,
                query_mask=query_mask,
                src_key_padding_mask=src_key_padding_mask,
            )

        output1 = output1.transpose(0, 1)
        # [b, len_ques + len_para, d_hid]
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

    def __init__(
        self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"
    ):
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
        if "activation" not in state:
            state["activation"] = F.relu
        super(CustomTransEncLayer, self).__setstate__(state)

    def forward(
        self,
        query: Tensor,
        key_value: Tensor,
        query_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Pass the input through the enc layer.

        Args:
            src: the sequence to the enc layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Trans class.
        """
        src2 = self.self_attn(
            query,
            key_value,
            key_value,
            attn_mask=query_mask,
            key_padding_mask=src_key_padding_mask,
        )[0]
        query = query + self.dropout1(src2)
        query = self.norm1(query)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
        query = query + self.dropout2(src2)
        query = self.norm2(query)
        return query, query


class MemoryBasedQuesRectifier(torch_nn.Module):
    def __init__(
        self,
        len_ques: int = 42,
        d_hid: int = 64,
        n_heads_trans: int = 8,
        n_layers_trans: int = 8,
        device: Any = None,
    ):
        super().__init__()

        self.d_hid = d_hid

        self.memory = Parameter(
            torch.zeros((len_ques, d_hid), device=device),
            requires_grad=True,
        )

        self.trans_enc = CustomTransEnc(
            enc_layer=CustomTransEncLayer(d_model=d_hid, nhead=n_heads_trans),
            num_layers=n_layers_trans,
            d_hid=d_hid,
        )

        self.ff1 = torch_nn.Sequential(torch_nn.Linear(d_hid, d_hid), torch_nn.Tanh())
        self.lin1 = torch_nn.Linear(d_hid, len_ques)
        self.lin2 = torch_nn.Linear(d_hid, d_hid)
        self.lin3 = torch_nn.Linear(d_hid, d_hid)

    def forward(self, ques: torch.Tensor, context: torch.Tensor):
        # ques  : [b, len_ques, d_hid]
        # contex: [b, n_paras, len_para, d_hid]
        b, len_ques, _ = ques.size()

        ###################################
        # Retrieve rectifier from memory
        ###################################

        context = context.view(b, -1, self.d_hid).mean(dim=1, keepdim=True)
        # [b, 1, d_hid]

        if self.memory.sum() != 0:
            rectifier = []
            for i in range(b):
                rectifier.append(
                    self.trans_enc(
                        query=context[i].unsqueeze(0),
                        key_value=self.memory.unsqueeze(0),
                    )
                )
            rectifier = torch.vstack(rectifier)
            # [b, 1, d_hid]
        else:
            rectifier = context
            # [b, 1, d_hid]
        rectifier = self.lin1(self.ff1(rectifier))
        # [b, 1, len_ques]
        rectifier = torch.softmax(rectifier.transpose(1, 2), dim=1)
        # [b, len_ques, 1]

        rectified_ques = rectifier * ques
        # [b, len_ques, d_hid]

        ###################################
        # Update memory
        ###################################

        context = self.lin2(context.repeat(1, len_ques, 1))
        # [b, len_ques, d_hid]
        ques = self.lin3(ques)
        # [b, len_ques, d_hid]

        gate = torch.sigmoid(context + ques)
        # [b, len_ques, d_hid]

        new_mem = self.memory
        for i in range(b):
            if new_mem.sum() != 0:
                new_mem = gate[i] * new_mem + (1 - gate[i]) * rectified_ques[i]
            else:
                new_mem = rectified_ques[i]

        self.memory = Parameter(new_mem.detach(), requires_grad=True)

        return rectified_ques


class MemoryBasedContextRectifier(torch_nn.Module):
    def __init__(
        self,
        len_para: int = 170,
        n_paras: int = 5,
        d_hid: int = 64,
        n_heads_trans: int = 8,
        n_layers_trans: int = 8,
        device: Any = None,
    ):
        super().__init__()

        self.d_hid = d_hid
        self.len_context = n_paras * len_para

        self.memory = Parameter(
            torch.zeros((self.len_context, d_hid), device=device),
            requires_grad=True,
        )

        self.trans_enc = CustomTransEnc(
            enc_layer=CustomTransEncLayer(d_model=d_hid, nhead=n_heads_trans),
            num_layers=n_layers_trans,
            d_hid=d_hid,
        )

        self.ff1 = torch_nn.Sequential(torch_nn.Linear(d_hid, d_hid), torch_nn.Tanh())
        self.lin1 = torch_nn.Linear(d_hid, self.len_context)
        self.lin2 = torch_nn.Linear(d_hid, d_hid)
        self.lin3 = torch_nn.Linear(d_hid, d_hid)

    def forward(self, ques: torch.Tensor, context: torch.Tensor):
        # ques  : [b, len_ques, d_hid]
        # contex: [b, n_paras, len_para, d_hid]

        b, n_paras, _, _ = context.size()

        ###################################
        # Retrieve rectifier from memory
        ###################################
        context = context.view(b, -1, self.d_hid)
        # [b, len_context, d_hid]

        ques = ques.mean(dim=1, keepdim=True)
        # [b, 1, d_hid]

        if self.memory.sum() != 0:
            rectifier = []
            for i in range(b):
                rectifier.append(
                    self.trans_enc(
                        query=ques[i].unsqueeze(0), key_value=self.memory.unsqueeze(0)
                    )
                )
            rectifier = torch.vstack(rectifier)
            # [b, 1, d_hid]
        else:
            rectifier = ques
            # [b, 1, d_hid]
        rectifier = self.lin1(self.ff1(rectifier))
        # [b, 1, len_context]
        rectifier = torch.softmax(rectifier.transpose(1, 2), dim=1)
        # [b, len_context, 1]

        ## Rectify context
        rectified_context = rectifier * context
        # [b, len_context, d_hid]

        ###################################
        # Update memory
        ###################################

        ques = self.lin2(ques.repeat(1, self.len_context, 1))
        # [b, len_context, d_hid]
        context = self.lin3(context)
        # [b, len_context, d_hid]

        gate = torch.sigmoid(context + ques)
        # [b, len_context, d_hid]

        new_mem = self.memory
        for i in range(b):
            if new_mem.sum() != 0:
                new_mem = gate[i] * new_mem + (1 - gate[i]) * rectified_context[i]
            else:
                new_mem = rectified_context[i]

        self.memory = Parameter(new_mem.detach(), requires_grad=True)

        return rectified_context.view(b, n_paras, -1, self.d_hid), rectifier.view(
            b, n_paras, -1
        )
