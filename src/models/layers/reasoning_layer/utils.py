from typing import Optional
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
import torch.nn as torch_nn


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
        d_hid: int = 64,
        n_heads_trans: int = 8,
        n_layers_trans: int = 8,
        norm=None,
    ):
        super(CustomTransEnc, self).__init__()
        self.layers = _get_clones(
            CustomTransEncLayer(d_model=d_hid, nhead=n_heads_trans), n_layers_trans
        )
        self.num_layers = n_layers_trans
        self.norm = norm

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
        # [b, len_, d_hid]

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
