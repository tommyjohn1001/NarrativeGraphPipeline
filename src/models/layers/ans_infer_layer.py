import random

from transformers import BertConfig, BertModel
import torch.nn as torch_nn
import torch.nn.functional as torch_f
import torch

from src.models.layers.bertbasedembd_layer import BertBasedEmbedding


class BertDecoder(torch_nn.Module):
    def __init__(
        self,
        seq_len_ans: int = 42,
        d_bert: int = 768,
        d_vocab: int = 30522,
        cls_tok_id: int = 101,
        embd_layer: torch.nn.Module = None,
    ):
        super().__init__()

        self.seq_len_ans = seq_len_ans
        self.d_bert = d_bert
        self.cls_tok_id = cls_tok_id

        self.embd_layer: BertBasedEmbedding = embd_layer

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
        # ans       : [b, seq_len, d_bert]
        # ans_mask  : [b, seq_len]

        seq_len = ans.shape[1]

        ans = torch_f.pad(ans, (0, 0, 0, self.seq_len_ans - seq_len), "constant", 0)
        ans_mask = torch_f.pad(ans_mask, (0, self.seq_len_ans - seq_len), "constant", 0)

        output = self.decoder(
            inputs_embeds=ans, attention_mask=ans_mask, encoder_hidden_states=Y
        )[0]
        # [b, seq_len_ans, 768]

        return output[:, :seq_len, :]

    def train(
        self, Y: torch.Tensor, ans: torch.Tensor, ans_mask, teacher_forcing_ratio: float
    ):
        # Y         : [b, seq_len_ans, d_bert]
        # ans       : [b, seq_len, d_bert]
        # ans_mask  : [b, seq_len]
        b, seq_len, _ = ans.shape

        input_embds = torch.full((b, 1), self.cls_tok_id, device=Y.device)
        input_embds = self.embd_layer.encode_ans(
            input_embds, ans_mask[:, : input_embds.shape[1]]
        ).detach()
        # [b, 1, d_bert]

        for _ in range(1, seq_len + 1):
            output = self(Y, input_embds, ans_mask[:, : input_embds.shape[1]])
            # [b, seq_len, d_bert]

            chosen = self.choose_teacher_forcing(
                teacher_forcing_ratio=teacher_forcing_ratio,
                output=output,
                ans=ans,
            )
            # [b, 1, 768]
            input_embds = torch.cat((input_embds, chosen.detach()), dim=1)

        pred = self.ff(output)
        # [b, seq_len_ans, d_vocab]

        return pred

    def choose_teacher_forcing(
        self, teacher_forcing_ratio: float, output: torch.Tensor, ans: torch.Tensor
    ):
        seq_len = output.shape[1]
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        seq_len = output.shape[1]

        if use_teacher_forcing:
            return ans[:, seq_len, :].unsqueeze(1)

        return output[:, -1, :].unsqueeze(1)

    def predict(self, Y: torch.Tensor, ans: torch.Tensor, ans_mask):
        # Y         : [b, seq_len_ans, d_bert]
        # ans       : [b, seq_len, d_bert]
        # ans_mask  : [b, seq_len]

        seq_len = ans.shape[1]

        ans = torch_f.pad(ans, (0, 0, 0, self.seq_len_ans - seq_len), "constant", 0)
        ans_mask = torch_f.pad(ans_mask, (0, self.seq_len_ans - seq_len), "constant", 0)

        output = self.decoder(
            inputs_embeds=ans, attention_mask=ans_mask, encoder_hidden_states=Y
        )[0]
        # [b, seq_len_ans, 768]

        pred = self.ff(output)
        # [b, seq_len_ans, d_vocab]

        pred = pred[:, :seq_len, :]

        return pred
