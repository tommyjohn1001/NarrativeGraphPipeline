from typing import Any


from transformers import BertConfig, BertModel
from torch.nn.parameter import Parameter
import torch.nn as torch_nn
import torch
import numpy as np
from transformers.models.bert.tokenization_bert import BertTokenizer


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
        self.epsilon = 1
        self.bigEPSILON = 0.5

        self.t = 0
        self.r = 0

        self.embd_layer: BertBasedEmbedding = embd_layer
        bert_conf = BertConfig()
        bert_conf.is_decoder = True
        bert_conf.hidden_size = d_hid
        bert_conf.add_cross_attention = True
        bert_conf.num_attention_heads = 8
        bert_conf.num_hidden_layers = 6
        self.trans_decoder = BertModel(config=bert_conf)

        self.lin1 = torch_nn.Linear(d_hid, d_vocab)

    def forward(
        self,
        Y: torch.Tensor,
        input_embds: torch.Tensor,
        input_masks: torch.Tensor,
    ):
        # Y          : [b, len_para, d_hid]
        # input_embds: [b, len_, d_vocab]
        # input_masks: [b, len_]

        # [b, d_hid]

        output = self.trans_decoder(
            inputs_embeds=input_embds,
            attention_mask=input_masks,
            encoder_hidden_states=Y,
        )[0]
        # [b, len_, d_hid]

        output = self.lin1(output)
        # [b, len_, d_vocab]

        output = torch.softmax(output, dim=-1)

        return output

    def get_groundtruth(self, b: int, ids: Any, output=None):
        # output: [b: d_vocab]

        #########################
        # Get distribution over vocab by token ids
        #########################
        mask = torch.zeros((b, self.d_vocab), device=self.device, requires_grad=False)
        # [b, d_vocab]

        if not torch.is_tensor(ids):
            assert isinstance(ids, int)
            ids = torch.full(
                (b, 1), fill_value=ids, device=self.device, requires_grad=False
            )
        ones = torch.ones((b, 1), device=self.device)
        mask = torch.scatter_add(mask, dim=1, index=ids, src=ones)
        # [b, d_vocab]

        #########################
        # Get word embedding
        #########################
        new_word = self.embd_layer.w_sum_ans(mask)
        # [b, d_hid]

        return output * mask if torch.torch.is_tensor(output) else None, new_word

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

        input_emds = []
        final = []

        ## Init input_embs with cls embedding
        _, cls_embd = self.get_groundtruth(b, self.tokenizer.cls_token_id)
        # [b, d_hid]

        input_emds.append(cls_embd.unsqueeze(1))

        for ith in range(1, self.len_ans + 1):
            output = self(
                Y=Y,
                input_embds=torch.cat(input_emds, dim=1),
                input_masks=ans_mask[:, :ith],
            )
            # [b, ith, d_vocab]

            ## Apply WEAM
            spare_p, new_word = self.choose_scheduled_sampling(
                output=output,
                ans_ids=ans_ids,
                cur_step=cur_step,
                max_step=max_step,
                cur_epoch=cur_epoch,
            )
            # spare_p: [b, d_vocab]
            # new_word: [b, d_hid]

            final.append(spare_p.unsqueeze(1))
            input_emds.append(new_word.unsqueeze(1))

        return torch.cat(final, dim=1).transpose(1, 2)

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
        output = output[:, -1, :]
        # [b, d_vocab]

        ## Create tensor 'pad_embd'

        ## If not exceed 15 epochs, keep using Teacher Forcing
        if cur_epoch < 15:
            if ith < self.len_ans:
                return self.get_groundtruth(b, ans_ids[:, ith].unsqueeze(1), output)

            return self.get_groundtruth(b, self.tokenizer.pad_token_id, output)

        # Apply Scheduled Sampling
        t = np.random.binomial(1, cur_step / max_step)
        self.t = t
        if t == 0:
            if ith < self.len_ans:
                return self.get_groundtruth(b, ans_ids[:, ith].unsqueeze(1), output)

            return self.get_groundtruth(b, self.tokenizer.pad_token_id, output)

        #################
        ## Calculate eta
        #################
        r = np.random.binomial(1, max((cur_step / max_step, self.bigEPSILON)))
        self.r = r
        epsilon = torch.exp(torch.tensor(-self.epsilon))
        if r == 0:
            if ith < self.len_ans:
                output_ = []
                for b_ in range(b):
                    output_.append(output[b_, ans_ids[b_, ith]].view(1))
                output_ = torch.cat(output_, dim=0)
                # [b]
                ita = epsilon * output_
            else:
                ita = epsilon * output[:, self.pad_tok_id]
        else:
            ita = epsilon * torch.max(epsilon * output, dim=-1)[0]
        # ita: [b]

        #################
        ## Calculate mask
        #################
        mask = torch.where(
            output > ita.unsqueeze(1).repeat(1, self.d_vocab),
            torch.ones(output.size()).type_as(output),
            torch.zeros(output.size()).type_as(output),
        )
        # [b, d_vocab]

        #################
        ## Recalculate output
        #################
        # TODO: no mater teacher forcing is used or not, spare_p still have to be calculated
        spare_p = output * mask
        # [b, d_vocab]
        spare_p = spare_p / torch.sum(spare_p, dim=1).unsqueeze(1).repeat(
            1, self.d_vocab
        )
        # [b, d_vocab]
        new_word = self.embd_layer.w_sum_ans(output)
        # [b, d_hid]

        return spare_p, new_word

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

        input_emds = []
        final = []

        ## Init input_embs with cls embedding
        _, cls_embd = self.get_groundtruth(b, self.tokenizer.cls_token_id)
        # [b, d_hid]

        input_emds.append(cls_embd.unsqueeze(1))

        for ith in range(1, self.len_ans + 1):
            output = self(
                Y=Y,
                input_embds=torch.cat(input_emds, dim=1),
                input_masks=ans_mask[:, :ith],
            )[:, -1, :]
            # [b, ith, d_vocab]

            ## Apply WEAM
            epsilon = torch.exp(torch.tensor(-self.epsilon))
            ita = epsilon * torch.max(epsilon * output, dim=-1)[0]
            # ita: [b]

            #################
            ## Calculate mask
            #################
            mask = torch.where(
                output > ita.unsqueeze(1).repeat(1, self.d_vocab),
                torch.ones(output.size()).type_as(output),
                torch.zeros(output.size()).type_as(output),
            )
            # [b, d_vocab]

            #################
            ## Recalculate output
            #################
            output = output * mask
            # [b, d_vocab]
            spare_p = output / torch.sum(output, dim=1).unsqueeze(1).repeat(
                1, self.d_vocab
            )
            # [b, d_vocab]

            _, topi = torch.topk(spare_p, k=1)
            # [b, 1]
            _, new_word = self.get_groundtruth(b, topi)
            # [b, d_hid]

            final.append(spare_p.unsqueeze(1))
            input_emds.append(new_word.unsqueeze(1))

        return torch.cat(final, dim=1).transpose(1, 2)
