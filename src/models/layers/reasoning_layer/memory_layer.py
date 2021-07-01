from typing import Any

import torch.nn.functional as torch_f
import torch.nn as torch_nn
import torch

from src.models.layers.reasoning_layer.utils import (
    MemoryBasedContextRectifier,
    MemoryBasedQuesRectifier,
)


class MemoryBasedReasoning(torch_nn.Module):
    def __init__(
        self,
        len_ques: int = 42,
        len_para: int = 500,
        n_paras: int = 3,
        n_heads_trans: int = 4,
        n_layers_trans: int = 3,
        d_hid: int = 64,
        device: Any = "cpu",
    ):
        super().__init__()

        self.len_ques = len_ques
        self.len_para = len_para
        self.n_paras = n_paras
        self.d_hid = d_hid

        self.ques_rectifier = MemoryBasedQuesRectifier(
            len_ques=len_ques,
            d_hid=d_hid,
            n_heads_trans=n_heads_trans,
            n_layers_trans=n_layers_trans,
            device=device,
        )

        self.contx_rectifier = MemoryBasedContextRectifier(
            len_para=len_para,
            n_paras=n_paras,
            d_hid=d_hid,
            n_heads_trans=n_heads_trans,
            n_layers_trans=n_layers_trans,
            device=device,
        )

        self.ff1 = torch_nn.Sequential(
            torch_nn.Linear(d_hid, d_hid),
            torch_nn.Linear(d_hid, d_hid),
            torch_nn.Tanh(),
            torch_nn.BatchNorm1d(len_ques),
        )
        self.ff2 = torch_nn.Sequential(
            torch_nn.Linear(d_hid, d_hid),
            torch_nn.Linear(d_hid, d_hid),
            torch_nn.Tanh(),
            torch_nn.BatchNorm1d(len_para),
        )
        self.lin2 = torch_nn.Linear(d_hid, d_hid)
        self.lin3 = torch_nn.Linear(d_hid * 2, d_hid)

    def forward(self, ques, context):
        # ques: [b, len_ques, d_hid]
        # context: [b, n_paras, len_para, d_hid]

        b, n_paras, len_para, _ = context.size()

        ###################################
        # Rectìfy question and context
        ###################################
        rect_ques = self.ques_rectifier(ques=ques, context=context)
        # [b, len_ques, d_hid]
        rect_context, contx_rectifier = self.contx_rectifier(ques=ques, context=context)
        # rect_context: [b, n_paras, len_para, d_hid]
        # contx_rectifier: [b, n_paras, len_para]

        ###################################
        # Use CoAttention to capture
        ###################################
        contx_rectifier = torch.max(contx_rectifier, dim=2)[0]
        # [b, n_paras]
        contx_rectifier = torch_f.gumbel_softmax(
            contx_rectifier, tau=1, hard=True, dim=1
        )
        # [b, n_paras]
        contx_rectifier = contx_rectifier.view(b, n_paras, 1, 1).repeat(
            1, 1, len_para, self.d_hid * 2
        )

        paras = []
        for ith in range(n_paras):
            para = rect_context[:, ith]
            # [b, len_para, d_hid]

            A_i = torch.bmm(
                self.ff1(rect_ques), self.lin2(self.ff2(para)).transpose(1, 2)
            )
            # [b, len_ques, len_para]

            A_q = torch.softmax(A_i, dim=1)
            A_d = torch.softmax(A_i, dim=2)

            C_q = torch.bmm(A_q, para)
            # [b, len_ques, d_hid]

            para = torch.bmm(A_d.transpose(1, 2), torch.cat((C_q, rect_ques), dim=-1))
            # [b, len_para, d_hid * 2]

            paras.append(para.unsqueeze(1))

        paras = torch.cat(paras, dim=1)
        # [b, n_paras, len_para, d_hid * 2]

        Y = (contx_rectifier * paras).sum(1)
        # [b, len_para, d_hid * 2]

        Y = self.lin3(Y)
        # [b, len_para, d_hid]

        return Y
