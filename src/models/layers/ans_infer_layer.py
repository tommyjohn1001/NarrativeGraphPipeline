import random

import torch.nn as torch_nn
import torch.nn.functional as torch_f
import torch

from src.models.layers.bertbasedembd_layer import BertBasedEmbedding


class BertDecoder(torch_nn.Module):
    def __init__(
        self,
        seq_len_ans: int = 42,
        n_layers_gru: int = 4,
        d_bert: int = 768,
        d_vocab: int = 30522,
        d_hid: int = 64,
        cls_tok_id: int = 101,
        embd_layer: torch.nn.Module = None,
    ):
        super().__init__()

        self.seq_len_ans = seq_len_ans
        self.d_bert = d_bert
        self.d_hid = d_hid
        self.cls_tok_id = cls_tok_id
        self.d_vocab = d_vocab

        self.embd_layer: BertBasedEmbedding = embd_layer
        self.d_hid_ = d_hid * 4

        self.gru = torch_nn.GRU(
            input_size=d_bert,
            hidden_size=d_hid * 2,
            num_layers=n_layers_gru,
            batch_first=True,
            bidirectional=True,
        )
        self.lin1 = torch_nn.Linear(self.d_hid_, self.d_hid_)
        self.ff1 = torch_nn.Sequential(
            torch_nn.Linear(self.d_hid_ * 2, self.d_hid_ * 2, bias=False),
            torch_nn.Tanh(),
            torch_nn.BatchNorm1d(seq_len_ans),
        )
        self.lin2 = torch_nn.Linear(self.d_hid_ * 2, d_vocab, bias=False)

        self.lin3 = torch_nn.Linear(1, d_vocab, bias=False)

    def forward(self, Y: torch.Tensor, ans: torch.Tensor, ans_mask: torch.Tensor):
        # Y        : [b, n_paras, seq_len_para, d_hid_]
        # ans      : [b, seq_len]
        # ans_mask : [b, seq_len]

        b, seq_len = ans.shape

        ans = torch_f.pad(ans, (0, self.seq_len_ans - seq_len), "constant", 0)
        ans_mask = torch_f.pad(ans_mask, (0, self.seq_len_ans - seq_len), "constant", 0)

        ##########################################
        ## Embed with BertBasedEmbd layer
        ##########################################
        ans = self.embd_layer.encode_ans(ans, ans_mask)
        # [b, seq_len_ans, d_bert]

        ##########################################
        ## Start decoding with GRU
        ##########################################
        ans_len = torch.sum(ans_mask, dim=1)
        tmp = torch_nn.utils.rnn.pack_padded_sequence(
            ans, ans_len, batch_first=True, enforce_sorted=False
        )
        tmp = self.gru(tmp)[0]
        d = torch_nn.utils.rnn.pad_packed_sequence(tmp, batch_first=True)[0]
        # [b, seq_len*, d_hid_]

        d = torch_f.pad(d, (0, 0, 0, self.seq_len_ans - d.shape[1]), "constant", 0)
        # [b, seq_len_ans, d_hid_]

        ##########################################
        ## Calculate generating score
        ##########################################
        Y = Y.reshape(b, -1, self.d_hid_)
        # [b, seq_len_contx = n_paras*seq_len_para, d_hid_]

        r = torch.bmm(Y, self.lin1(d).transpose(1, 2))
        # [b, seq_len_contx, seq_len_ans]

        a = torch.softmax(r, dim=1)

        c = torch.bmm(a.transpose(1, 2), Y)
        # [b, seq_len_ans, d_hid_]

        d_ = self.ff1(torch.cat((d, c), dim=2))
        # [b, seq_len_ans, d_hid_ * 2]

        score_gen = self.lin2(d_)
        # [b, seq_len_ans, d_vocab]

        ##########################################
        ## Calculate pointing score
        ##########################################
        score_copy = torch.max(r, dim=1)[0]
        # [b, seq_len_ans]

        score_copy = self.lin3(score_copy.unsqueeze(2))
        # [b, seq_len_ans, d_vocab]

        ##########################################
        ## Combine copy and pointing score
        ##########################################
        total = torch.cat((score_gen, score_copy), dim=2)
        # [b, seq_len_ans, d_vocab * 2]

        total = torch.softmax(total, dim=-1)

        pred = total[:, :, : self.d_vocab] + total[:, :, self.d_vocab :]
        # [b, seq_len_ans, d_vocab]

        return pred[:, :seq_len, :]

    def do_train(
        self,
        Y: torch.Tensor,
        ans: torch.Tensor,
        ans_mask: torch.Tensor,
        teacher_forcing_ratio: float,
    ):
        # Y         : [b, n_paras, seq_len_para, d_hid_]
        # ans       : [b, seq_len_ans]
        # ans_mask  : [b, seq_len_ans]
        b = ans.shape[0]

        input_ids = torch.full((b, 1), self.cls_tok_id, device=Y.device)
        # [b, 1]

        for ith in range(1, self.seq_len_ans + 1):
            output = self(Y, input_ids, ans_mask[:, : input_ids.shape[1]])
            # [b, ith, d_vocab]

            if ith == self.seq_len_ans:
                break

            output = torch.argmax(output, dim=2)
            chosen = self.choose_teacher_forcing(
                teacher_forcing_ratio=teacher_forcing_ratio,
                output=output,
                ans=ans,
            )
            # [b]
            input_ids = torch.cat((input_ids, chosen.detach().unsqueeze(1)), dim=1)

        return output

    def choose_teacher_forcing(
        self, teacher_forcing_ratio: float, output: torch.Tensor, ans: torch.Tensor
    ):
        # output: [b, seq_len]
        # ans   : [b, seq_len_ans]
        seq_len = output.shape[1]
        use_teacher_forcing = random.random() < teacher_forcing_ratio

        if use_teacher_forcing:
            return ans[:, seq_len]

        return output[:, seq_len]
