from typing import Any


from transformers import BertConfig, BertModel
from transformers.models.bert.tokenization_bert import BertTokenizer
import torch.nn as torch_nn
import torch
import numpy as np


from src.models.layers.finegrain_layer import FineGrain


class Decoder(torch_nn.Module):
    def __init__(
        self,
        len_ans: int = 15,
        d_vocab: int = 30522,
        d_bert: int = 768,
        tokenizer: BertTokenizer = None,
        embd_layer: torch.nn.Module = None,
    ):
        super().__init__()

        self.len_ans = len_ans
        self.tokenizer = tokenizer
        self.d_vocab = d_vocab
        self.device = None
        self.epsilon = 1
        self.bigEPSILON = 0.5

        self.t = 0
        self.r = 0

        self.embd_layer: FineGrain = embd_layer
        bert_conf = BertConfig()
        bert_conf.is_decoder = True

        bert_conf.add_cross_attention = True
        bert_conf.num_attention_heads = 8
        bert_conf.num_hidden_layers = 6
        self.trans_decoder = BertModel(config=bert_conf)

        self.lin1 = torch_nn.Sequential(
            torch_nn.Linear(d_bert, d_bert),
            torch_nn.Tanh(),
            torch_nn.Linear(d_bert, d_vocab),
        )

    ##############################################
    # Methods for training
    ##############################################

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
        masks = []

        for ith in range(1, self.len_ans):
            ##################
            # Choose next word for decoding
            # based on Scheduled Sampling
            ##################
            output = self(
                Y=Y,
                input_embds=torch.cat(input_emds, dim=1),
                input_masks=ans_mask[:, :ith],
            )
            # [b, ith, d_vocab]

            ##################
            # Choose next word for decoding
            # based on Scheduled Sampling
            ##################
            mask = self.get_mask(
                output, ans_ids=ans_ids, cur_step=cur_step, max_step=max_step
            )
            # [b, d_vocab]
            masks.append(mask.unsqueeze(1))

            ##################
            # Choose next word for decoding
            # based on Scheduled Sampling
            ##################
            if ith < self.len_ans - 1:
                ## Apply WEAM
                new_word = self.choose_scheduled_sampling(
                    output=output,
                    ans_ids=ans_ids,
                    cur_step=cur_step,
                    max_step=max_step,
                    cur_epoch=cur_epoch,
                )
                # new_word: [b, d_bert]

                input_emds.append(new_word.unsqueeze(1))

        ## Get final mask and apply to output
        mask = torch.cat(masks, dim=1)
        # [b, len_ans - 1, d_vocab]

        output_sum = (
            (output * mask).sum(dim=-1, keepdim=True).repeat(1, 1, self.d_vocab)
        )
        # [b, len_ans - 1, d_vocab]

        output = output / output_sum
        # [b, len_ans - 1, d_vocab]

        ## Transpose to match with loss function
        output = output.transpose(1, 2)
        # [b, d_vocab, len_ans - 1]

        return output

    def get_mask(
        self,
        output: torch.Tensor,
        ans_ids: torch.Tensor,
        cur_step: int,
        max_step: int,
    ):
        # output: [b, len_, d_vocab]
        # ans_ids: [b, len_ans]

        ans_ids = ans_ids[:, output.size(1) - 1].unsqueeze(-1)
        # [b, 1]
        output = output[:, -1, :]
        # [b, d_vocab]

        # Calculate eta
        self.r = np.random.binomial(1, max((cur_step / max_step, self.bigEPSILON)))
        eps = torch.exp(torch.tensor(-self.epsilon))
        if self.r == 0:
            ita = eps * output.gather(dim=-1, index=ans_ids)
        else:
            ita = eps * torch.max(eps * output, dim=-1)[0].unsqueeze(-1)
        # ita: [b, 1]

        # Calculate mask
        mask = torch.where(
            output > ita.repeat(1, self.d_vocab),
            torch.ones(output.size()).type_as(output),
            torch.zeros(output.size()).type_as(output),
        )
        # [b, d_vocab]

        return mask

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
            else self.embd_layer.get_w_embd(
                input_embds=torch.softmax(output[:, -1, :], dim=-1)
            )
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
                _, topi = torch.topk(output, k=1)
                # [b, 1]
                new_word = self.get_new_word(b, topi[:, -1])

                input_emds.append(new_word)

        output = output.transpose(1, 2)
        # [b, d_vocab, len_ans - 1]

        return output
