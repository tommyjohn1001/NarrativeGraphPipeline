import random

from transformers import BertConfig, BertModel
import torch.nn as torch_nn
import torch.nn.functional as torch_f
import torch

from src.models.layers.finegrain_layer import FineGrain


class BertDecoder(torch_nn.Module):
    def __init__(
        self,
        seq_len_ans: int = 15,
        d_bert: int = 768,
        d_vocab: int = 30552,
        cls_tok_id: int = 101,
        embd_layer: torch.nn.Module = None,
    ):
        super().__init__()

        self.cls_tok_id = cls_tok_id
        self.seq_len_ans = seq_len_ans

        self.embd_layer: FineGrain = embd_layer

        bert_conf = BertConfig()
        bert_conf.is_decoder = True
        bert_conf.add_cross_attention = True
        bert_conf.num_attention_heads = 6
        bert_conf.num_hidden_layers = 6
        self.decoder = BertModel(config=bert_conf)

        self.ff = torch_nn.Sequential(
            torch_nn.Linear(d_bert, d_bert),
            torch_nn.GELU(),
            torch_nn.Linear(d_bert, d_vocab),
        )

    def forward(self, Y: torch.Tensor, ans_ids: torch.Tensor, ans_mask: torch.Tensor):
        # Y       : [b, seq_len_ans, d_bert]
        # ans_ids : [b, seq_len]
        # ans_mask: [b, seq_len]

        input_embds = self.embd_layer.encode_ans(ans_ids=ans_ids, ans_mask=ans_mask)

        output = self.decoder(
            inputs_embeds=input_embds, attention_mask=ans_mask, encoder_hidden_states=Y
        )[0]
        # [b, seq_len_ans, 768]

        pred = self.ff(output)
        # [b, seq_len_ans, d_vocab]

        return pred

    def do_train(
        self,
        Y: torch.Tensor,
        ans_ids: torch.Tensor,
        ans_mask: torch.Tensor,
        teacher_forcing_ratio: float,
    ):
        # Y         : [b, len_para, d_hid]
        # ans_ids   : [b, len_ans]
        # ans_mask  : [b, len_ans]

        input_ids = torch.full((Y.size()[0], 1), self.cls_tok_id, device=Y.device)
        # [b, 1]

        for ith in range(1, self.seq_len_ans + 1):
            output = self(Y=Y, ans_ids=input_ids, ans_mask=ans_mask[:, :ith])
            # [b, ith, d_vocab]

            ## Apply Scheduling teacher
            if ith == self.seq_len_ans:
                break

            # _, topi = torch.topk(output[:, -1, :], k=1)
            topi = torch.argmax(output[:, -1, :], dim=1, keepdim=True)
            chosen = self.choose_teacher_forcing(
                teacher_forcing_ratio=teacher_forcing_ratio,
                output=topi,
                ans_ids=ans_ids,
                ith=ith,
            )
            # [b]
            input_ids = torch.cat((input_ids, chosen.detach()), dim=1)

        return output

    def choose_teacher_forcing(
        self,
        teacher_forcing_ratio: float,
        output: torch.Tensor,
        ans_ids: torch.Tensor,
        ith: int,
    ):
        # output: [b, 1]
        # ans_ids   : [b, len_ans]
        use_teacher_forcing = random.random() < teacher_forcing_ratio

        return ans_ids[:, ith].unsqueeze(1) if use_teacher_forcing else output
