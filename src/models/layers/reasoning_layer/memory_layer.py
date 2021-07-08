import torch.nn as torch_nn
import torch

from src.models.layers.reasoning_layer.utils import CustomTransEnc


class MemoryBasedReasoning(torch_nn.Module):
    def __init__(
        self,
        len_ques: int = 42,
        len_para: int = 170,
        n_heads_trans: int = 4,
        n_layers_trans: int = 3,
        d_hid: int = 64,
    ):
        super().__init__()

        self.len_ques = len_ques
        self.len_para = len_para
        self.d_hid = d_hid

        self.trans_enc_contx = CustomTransEnc(
            n_heads_trans=n_heads_trans,
            n_layers_trans=n_layers_trans,
            d_hid=d_hid,
        )
        self.trans_enc_ans = CustomTransEnc(
            n_heads_trans=n_heads_trans,
            n_layers_trans=n_layers_trans,
            d_hid=d_hid,
        )

        self.ff1 = torch_nn.Sequential(
            torch_nn.Linear(d_hid, d_hid),
            torch_nn.BatchNorm1d(len_ques + len_para),
            torch_nn.Sigmoid(),
        )
        self.lin1 = torch_nn.Linear(d_hid, d_hid)
        self.lin2 = torch_nn.Linear(d_hid, d_hid)

        self.ff2 = torch_nn.Sequential(
            torch_nn.Linear(d_hid, d_hid),
            torch_nn.Sigmoid(),
        )
        self.lin3 = torch_nn.Linear(d_hid, d_hid)
        self.lin4 = torch_nn.Linear(d_hid, d_hid)

    def forward(self, ques, context, ans):
        # ques: [b, len_ques, d_hid]
        # context: [b, n_paras, len_para, d_hid]
        # ans: [b, len_, d_hid]

        mem_contx, mem_ans = None, None
        for ith in range(context.size(1)):
            ##################
            # Encode ques-para
            ##################
            ques_para = torch.cat((ques, context[:, ith]), dim=1)
            # [b, len_ques+len_para, d_hid]

            if not torch.is_tensor(mem_contx):
                mem_contx = ques_para
                # [b, len_ques+len_para, d_hid]

            output = self.trans_enc_contx(query=mem_contx, key_value=ques_para)
            # [b, len_ques+len_para, d_hid]

            gate_contx = self.ff1(self.lin1(mem_contx) + self.lin2(output))
            # [b, len_ques+len_para, d_hid]

            mem_contx = gate_contx * mem_contx + (1 - gate_contx) * ques_para
            # [b, len_ques+len_para, d_hid]

            ##################
            # Encode ans
            ##################
            if not torch.is_tensor(mem_ans):
                mem_ans = ans
                # [b, len_, d_hid]

            output = self.trans_enc_ans(query=ans, key_value=mem_contx)
            # [b, len_, d_hid]

            gate_ans = self.ff2(self.lin3(output) + self.lin4(ans))
            # [b, len_, d_hid]

            mem_ans = gate_ans * ans + (1 - gate_ans) * mem_ans

        return mem_ans
        # [b, len_, d_hid]
