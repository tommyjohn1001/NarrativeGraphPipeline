from transformers import BertConfig, BertModel
import torch.nn as torch_nn
import torch.nn.functional as torch_f
import torch


class BertDecoder(torch_nn.Module):
    def __init__(
        self,
        seq_len_ans: int = 15,
        d_bert: int = 768,
        d_vocab: int = 30552,
    ):
        super().__init__()

        self.seq_len_ans = seq_len_ans

        bert_conf = BertConfig()
        bert_conf.is_decoder = True
        bert_conf.add_cross_attention = True
        bert_conf.num_attention_heads = 6
        bert_conf.num_hidden_layers = 6
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

        output = self.decoder(
            inputs_embeds=ans, attention_mask=ans_mask, encoder_hidden_states=Y
        )[0]
        # [b, seq_len_ans, 768]

        pred = self.ff(output).transpose(1, 2)
        # [b, d_vocab, seq_len_ans]

        return pred
