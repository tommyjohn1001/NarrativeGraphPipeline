import random

from transformers import BertConfig, BertModel
import torch.nn as torch_nn
import torch
import numpy as np

from src.models.layers.finegrain_layer import FineGrain


class BertDecoder(torch_nn.Module):
    def __init__(
        self,
        len_ans: int = 15,
        d_bert: int = 768,
        d_vocab: int = 30552,
        cls_tok_id: int = 101,
        embd_layer: torch.nn.Module = None,
    ):
        super().__init__()

        self.cls_tok_id = cls_tok_id
        self.len_ans = len_ans
        self.t = -1

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
        # Y       : [b, len_ans, d_bert]
        # ans_ids : [b, seq_len]
        # ans_mask: [b, seq_len]

        input_embds = self.embd_layer.encode_ans(ans_ids=ans_ids, ans_mask=ans_mask)

        output = self.decoder(
            inputs_embeds=input_embds, attention_mask=ans_mask, encoder_hidden_states=Y
        )[0]
        # [b, len_ans, 768]

        pred = self.ff(output)
        # [b, len_ans, d_vocab]

        return pred

    def do_train(
        self,
        Y: torch.Tensor,
        ans_ids: torch.Tensor,
        ans_mask: torch.Tensor,
        cur_step: int,
        max_step: int,
        cur_epoch: int,
        is_valid: bool = False,
    ):
        # Y         : [b, len_para, d_hid]
        # ans_ids   : [b, len_ans]
        # ans_mask  : [b, len_ans]

        input_ids = torch.full((Y.size()[0], 1), self.cls_tok_id, device=Y.device)
        # [b, 1]

        for ith in range(1, self.len_ans + 1):
            output = self(Y=Y, ans_ids=input_ids, ans_mask=ans_mask[:, :ith])
            # [b, ith, d_vocab]

            ## Apply Scheduling teacher
            if ith == self.len_ans:
                break

            _, topi = torch.topk(output[:, -1, :], k=1)
            if is_valid:
                chosen = topi
            else:
                chosen = self.choose_scheduled_sampling(
                    output=topi,
                    ans_ids=ans_ids,
                    ith=ith,
                    cur_step=cur_step,
                    max_step=max_step,
                    cur_epoch=cur_epoch,
                )
            # [b]
            input_ids = torch.cat((input_ids, chosen.detach()), dim=1)

        return output.transpose(1, 2)

    def choose_scheduled_sampling(
        self,
        output: torch.Tensor,
        ans_ids: torch.Tensor,
        ith: int,
        cur_step: int,
        max_step: int,
        cur_epoch: int,
    ):

        # output: [b, 1]
        # ans_ids   : [b, len_ans]
        if cur_epoch < 15:
            t = 0

        if cur_epoch < 20:
            t = np.random.binomial(1, min((cur_step / max_step, 0.5)))
        elif cur_epoch < 25:
            t = np.random.binomial(1, min((cur_step / max_step, 0.75)))
        else:
            t = np.random.binomial(1, cur_step / max_step)

        self.t = t

        return ans_ids[:, ith].unsqueeze(1) if t == 0 else output
