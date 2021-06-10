from transformers import BertConfig, BertModel
import torch.nn as torch_nn
import torch.nn.functional as torch_f
import torch


class BertDecoder(torch_nn.Module):
    def __init__(
        self,
        seq_len_ans: int = 42,
        d_bert: int = 768,
        d_vocab: int = 30552,
    ):
        super().__init__()

        self.seq_len_ans = seq_len_ans

        bert_conf = BertConfig()
        bert_conf.is_decoder = True
        bert_conf.add_cross_attention = True
        # NOTE: This is the difference with Thong's model: Thong loads from pretrain and freeze while I dont load and train
        self.decoder = BertModel(config=bert_conf)

        self.ff = torch_nn.Sequential(
            torch_nn.Linear(d_bert, d_bert),
            torch_nn.BatchNorm1d(seq_len_ans),
            torch_nn.GELU(),
            torch_nn.Linear(d_bert, d_vocab),
        )

    def forward(self, Y: torch.Tensor, ans: torch.Tensor, ans_mask):
        # Y         : [b, seq_len_ans, d_bert]
        # ans       : [b, seq_len]
        # ans_mask  : [b, seq_len]

        seq_len = ans.shape[1]

        ans = torch_f.pad(ans, (0, 0, 0, self.seq_len_ans - seq_len), "constant", 0)
        ans_mask = torch_f.pad(ans_mask, (0, self.seq_len_ans - seq_len), "constant", 0)

        output = self.decoder(
            input_ids=ans, attention_mask=ans_mask, encoder_hidden_states=Y
        )[0]
        # [b, seq_len_ans, 768]

        pred = self.ff(output)
        # [b, seq_len_ans, d_vocab]

        pred = pred[:, :seq_len, :]

        return pred
