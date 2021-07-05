from typing import Any


from transformers import BertConfig, BertModel
from transformers.models.bert.tokenization_bert import BertTokenizer
import torch.nn as torch_nn
import torch
import numpy as np


from src.models.layers.bertbasedembd_layer import BertBasedEmbedding


class Decoder(torch_nn.Module):
    def __init__(
        self,
        len_ans: int = 15,
        d_vocab: int = 30522,
        d_hid: int = 64,
        tokenizer: BertTokenizer = None,
        embd_layer: torch.nn.Module = None,
    ):
        super().__init__()

        self.len_ans = len_ans
        self.d_hid = d_hid
        self.tokenizer = tokenizer
        self.d_vocab = d_vocab
        self.device = None

        self.t = 0

        self.embd_layer: BertBasedEmbedding = embd_layer
        bert_conf = BertConfig()
        bert_conf.is_decoder = True
        bert_conf.hidden_size = d_hid
        bert_conf.add_cross_attention = True
        bert_conf.num_attention_heads = 8
        bert_conf.num_hidden_layers = 6
        self.trans_decoder = BertModel(config=bert_conf)

        self.lin1 = torch_nn.Sequential(
            torch_nn.Linear(d_hid, d_hid),
            torch_nn.Tanh(),
            torch_nn.Linear(d_hid, d_vocab),
        )

    def forward(
        self,
        Y: torch.Tensor,
        input_embds: torch.Tensor,
        input_masks: torch.Tensor,
    ):
        # Y          : [b, len_para, d_hid]
        # input_embds: [b, len_, d_vocab]
        # input_masks: [b, len_]

        encoded = self.embd_layer.encode_ans(
            input_embds=input_embds, input_masks=input_masks
        )

        output = self.trans_decoder(
            inputs_embeds=encoded,
            attention_mask=input_masks,
            encoder_hidden_states=Y,
        )[0]
        # [b, len_, d_hid]

        output = self.lin1(output)
        # [b, len_, d_vocab]

        # output = torch.softmax(output, dim=-1)

        return output

    def get_new_word(self, b: int, ids: Any):

        if not torch.is_tensor(ids):
            assert isinstance(ids, int)
            ids = torch.full(
                (b,), fill_value=ids, device=self.device, requires_grad=False
            )

        new_word = self.embd_layer.get_w_embd(input_ids=ids)
        # [b, d_bert]

        return new_word

    def do_train(
        self,
        Y: torch.Tensor,
        ans_ids: torch.Tensor,
        ans_mask: torch.Tensor,
        cur_step: int,
        max_step: int,
        cur_epoch: int,
    ):
        # Y         : [b, len_para, d_hid]
        # ans_ids   : [b, len_ans]
        # ans_mask  : [b, len_ans]

        b = Y.size(0)
        self.device = Y.device

        ## Init input_embs with cls embedding
        cls_embd = self.get_new_word(b, self.tokenizer.cls_token_id)
        # [b, d_hid]

        input_emds = [cls_embd.unsqueeze(1)]

        for ith in range(1, self.len_ans):
            output = self(
                Y=Y,
                input_embds=torch.cat(input_emds, dim=1),
                input_masks=ans_mask[:, :ith],
            )
            # [b, ith, d_vocab]

            if ith < self.len_ans - 1:
                ## Apply WEAM
                new_word = self.choose_scheduled_sampling(
                    output=torch.softmax(output, dim=-1),
                    ans_ids=ans_ids,
                    cur_step=cur_step,
                    max_step=max_step,
                    cur_epoch=cur_epoch,
                )
                # new_word: [b, d_bert]

                input_emds.append(new_word.unsqueeze(1))

        ## Get output for OT
        output_ot = self.embd_layer.get_output_ot(output)
        # [b, len_ans - 1, d_hid]

        output_mle = output.transpose(1, 2)
        # [b, d_vocab, len_ans - 1]

        return output_mle, output_ot

    ##############################################
    # Methods for scheduled sampling
    ##############################################

    def choose_scheduled_sampling(
        self,
        output: torch.Tensor,
        ans_ids: torch.Tensor,
        cur_step: int,
        max_step: int,
        cur_epoch: int,
    ):
        # output: [b, len_, d_vocab]
        # ans_ids: [b, len_ans]

        b = output.size(0)

        ith = output.size(1)

        # Apply Scheduled Sampling
        if cur_epoch < 20:
            t = np.random.binomial(1, min((cur_step / max_step, 0.5)))
        elif cur_epoch < 25:
            t = np.random.binomial(1, min((cur_step / max_step, 0.75)))
        else:
            t = np.random.binomial(1, cur_step / max_step)

        self.t = t

        new_word = (
            self.get_new_word(b, ans_ids[:, ith])
            if self.t == 0
            else self.embd_layer.get_w_embd(input_embds=output[:, -1, :])
        )
        # new_word: [b, d_bert]

        return new_word

    ##############################################
    # Methods for validation/prediction
    ##############################################
    def do_predict(
        self,
        Y: torch.Tensor,
        ans_mask: torch.Tensor,
    ):

        b = Y.size(0)
        self.device = Y.device

        ## Init input_embs with cls embedding
        cls_embd = self.get_new_word(b, self.tokenizer.cls_token_id)
        # [b, d_hid]

        input_emds = [cls_embd.unsqueeze(1)]

        for ith in range(1, self.len_ans):
            output = self(
                Y=Y,
                input_embds=torch.cat(input_emds, dim=1),
                input_masks=ans_mask[:, :ith],
            )
            # [b, ith, d_vocab]

            if ith < self.len_ans - 1:
                ## Apply WEAM
                _, topi = torch.topk(output, k=1)
                # [b, 1]
                new_word = self.get_new_word(b, topi[:, -1])

                input_emds.append(new_word)

        ## Get output for OT
        output_ot = self.embd_layer.get_output_ot(output)
        # [b, len_ans - 1, d_hid]

        output_mle = output.transpose(1, 2)
        # [b, d_vocab, len_ans - 1]

        return output_mle, output_ot
