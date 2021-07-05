from typing import Any, Optional
import json

from transformers import AdamW, BertTokenizer
import pytorch_lightning as plt
import torch.nn.functional as torch_F
import torch.nn as torch_nn
import torch


from src.datamodules.narrative_datamodule import NarrativeDataModule
from src.models.layers.reasoning_layer.memory_layer import MemoryBasedReasoning
from src.models.layers.bertbasedembd_layer import BertBasedEmbedding
from src.models.layers.ans_infer_layer import Decoder
from src.utils.generator import GeneratorOwn
from src.utils.utils import ipot, get_scores

EPSILON = 10e-10


class NarrativeModel(plt.LightningModule):
    def __init__(
        self,
        len_ques: int = 42,
        len_para: int = 170,
        len_ans: int = 15,
        n_paras: int = 5,
        n_layers_trans: int = 3,
        n_heads_trans: int = 4,
        d_hid: int = 64,
        d_bert: int = 768,
        d_vocab: int = 30522,
        lr: float = 1e-5,
        w_decay: float = 1e-2,
        switch_frequency: int = 5,
        beam_size: int = 20,
        n_gram_beam: int = 5,
        path_bert: str = None,
        path_pred: str = None,
        path_train_pred: str = None,
        datamodule: NarrativeDataModule = None,
    ):

        super().__init__()

        self.d_vocab = d_vocab
        self.lr = lr
        self.w_decay = w_decay
        self.beam_size = beam_size
        self.n_gram_beam = n_gram_beam
        self.switch_frequency = switch_frequency
        self.len_ans = len_ans

        self.path_pred = path_pred
        self.path_train_pred = path_train_pred

        self.bert_tokenizer = BertTokenizer.from_pretrained(path_bert)
        self.datamodule: NarrativeDataModule = datamodule

        #############################
        # Define model
        #############################
        self.embd_layer = BertBasedEmbedding(d_bert=d_bert, path_bert=path_bert)
        self.reasoning = MemoryBasedReasoning(
            len_ques=len_ques,
            len_para=len_para,
            n_paras=n_paras,
            n_heads_trans=n_heads_trans,
            n_layers_trans=n_layers_trans,
            d_hid=d_hid,
            device=self.device,
        )
        self.ans_infer = Decoder(
            len_ans=len_ans,
            d_vocab=d_vocab,
            d_hid=d_hid,
            tokenizer=self.bert_tokenizer,
            embd_layer=self.embd_layer,
        )

        ## Freeeze some parameters
        list_freeze_sets = [
            self.embd_layer.bert_emb.parameters(),
            # self.ans_infer.decoder.parameters(),
        ]
        for params in list_freeze_sets:
            for param in params:
                param.requires_grad = True

        #############################
        # Define things
        #############################
        self.criterion = torch_nn.CrossEntropyLoss(
            ignore_index=self.bert_tokenizer.pad_token_id
        )

    ####################################################################
    # FOR TRAINING PURPOSE
    ####################################################################

    def get_loss(self, output_mle, output_ot, ans_ids, ans_mask, gamma=0.1):
        # output_mle: [b, d_vocab, len_ans-1]
        # output_ot: [b, len_ans-1, d_hid]
        # ans_ids: [b, len_ans-1]
        # ans_mask: [b, len_ans-1]

        # Calculate MLE loss
        loss_mle = self.criterion(output_mle, ans_ids)

        # Calculate OT loss
        ans = self.embd_layer.encode_ans(input_ids=ans_ids, input_masks=ans_mask)
        # [b, len_ans-1, d_hid]

        loss_ot = ipot(output_ot, ans)

        total_loss = loss_mle + gamma * loss_ot

        return total_loss

    def get_prediction(self, output_mle, ans1_ids, ans2_ids):
        _, prediction = torch.topk(torch_F.log_softmax(output_mle, dim=1), 1, dim=1)

        prediction = [
            {
                "pred": " ".join(self.bert_tokenizer.convert_ids_to_tokens(pred_)),
                "ref": [
                    " ".join(self.bert_tokenizer.convert_ids_to_tokens(ans1_)),
                    " ".join(self.bert_tokenizer.convert_ids_to_tokens(ans2_)),
                ],
            }
            for pred_, ans1_, ans2_ in zip(prediction.squeeze(1), ans1_ids, ans2_ids)
        ]

        return prediction

    def model(
        self,
        ques_ids,
        ques_mask,
        ans_ids,
        ans_mask,
        context_ids,
        context_mask,
        cur_step: int = 0,
        max_step: int = 0,
        cur_epoch: int = 0,
        is_valid: bool = False,
    ):
        # ques_ids: [b, len_ques]
        # ques_mask: [b, len_ques]
        # context_ids: [b, n_paras, len_para]
        # context_mask: [b, n_paras, len_para]
        # ans_ids: [b, len_ans]
        # ans_mask: [b, len_ans]

        ####################
        # Embed question, context and answer
        ####################
        ques, context = self.embd_layer.encode_ques_para(
            ques_ids=ques_ids,
            context_ids=context_ids,
            ques_mask=ques_mask,
            context_mask=context_mask,
        )
        # ques : [b, len_ques, d_hid]
        # context: [b, n_paras, len_para, d_hid]

        ####################
        # Do reasoning
        ####################
        Y = self.reasoning(ques=ques, context=context)
        # [b, len_ans, d_hid]

        ####################
        # Generate answer
        ####################
        if is_valid:
            return self.ans_infer.do_predict(
                Y=Y,
                ans_mask=ans_mask,
            )

        return self.ans_infer.do_train(
            Y=Y,
            ans_ids=ans_ids,
            ans_mask=ans_mask,
            cur_step=cur_step,
            max_step=max_step,
            cur_epoch=cur_epoch,
        )
        # pred: [b, len_ans, d_vocab]

    def training_step(self, batch: Any, batch_idx: int):
        ques_ids = batch["ques_ids"]
        ques_mask = batch["ques_mask"]
        ans1_ids = batch["ans1_ids"]
        ans2_ids = batch["ans2_ids"]
        ans1_mask = batch["ans1_mask"]
        context_ids = batch["context_ids"]
        context_mask = batch["context_mask"]

        output_mle, output_ot = self.model(
            ques_ids=ques_ids,
            ques_mask=ques_mask,
            ans_ids=ans1_ids,
            ans_mask=ans1_mask,
            context_ids=context_ids,
            context_mask=context_mask,
            cur_step=batch_idx,
            max_step=self.datamodule.data_train.size_dataset
            // self.datamodule.batch_size,
            cur_epoch=self.current_epoch,
        )
        # output_ot: [b, len_ans - 1, d_hid]
        # output_mle: [b, d_vocab, len_ans - 1]

        loss = self.get_loss(output_mle, output_ot, ans1_ids[:, 1:], ans1_mask[:, 1:])
        prediction = self.get_prediction(output_mle, ans1_ids, ans2_ids)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log(
            "train/t", self.ans_infer.t, on_step=True, on_epoch=False, prog_bar=False
        )

        return {"loss": loss, "pred": prediction}

    def on_train_batch_end(
        self, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        with open(self.path_train_pred, "a+") as pred_file:
            json.dump(outputs["pred"], pred_file, indent=2, ensure_ascii=False)

    def on_train_epoch_end(self) -> None:
        if self.current_epoch % self.switch_frequency == 0 and self.current_epoch != 0:
            self.datamodule.switch_answerability()

    def test_step(self, batch: Any, batch_idx: int):
        ques_ids = batch["ques_ids"]
        ques_mask = batch["ques_mask"]
        ans1_ids = batch["ans1_ids"]
        ans1_mask = batch["ans1_mask"]
        context_ids = batch["context_ids"]
        context_mask = batch["context_mask"]

        output_mle, output_ot = self.model(
            ques_ids=ques_ids,
            ques_mask=ques_mask,
            ans_ids=ans1_ids,
            ans_mask=ans1_mask,
            context_ids=context_ids,
            context_mask=context_mask,
            cur_step=batch_idx,
            cur_epoch=self.current_epoch,
        )
        # output_ot: [b, len_ans - 1, d_hid]
        # output_mle: [b, d_vocab, len_ans - 1]

        loss = self.get_loss(output_mle, output_ot, ans1_ids[:, 1:], ans1_mask[:, 1:])

        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=False)

        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        ques_ids = batch["ques_ids"]
        ques_mask = batch["ques_mask"]
        ans1_ids = batch["ans1_ids"]
        ans2_ids = batch["ans2_ids"]
        ans1_mask = batch["ans1_mask"]
        context_ids = batch["context_ids"]
        context_mask = batch["context_mask"]

        output_mle, output_ot = self.model(
            ques_ids=ques_ids,
            ques_mask=ques_mask,
            ans_ids=ans1_ids,
            ans_mask=ans1_mask,
            context_ids=context_ids,
            context_mask=context_mask,
            is_valid=True,
        )
        # output_ot: [b, len_ans - 1, d_hid]
        # output_mle: [b, d_vocab, len_ans - 1]

        loss = self.get_loss(output_mle, output_ot, ans1_ids[:, 1:], ans1_mask[:, 1:])
        prediction = self.get_prediction(output_mle, ans1_ids, ans2_ids)

        self.log("valid/loss", loss, on_step=False, on_epoch=True, prog_bar=False)

        return {"loss": loss, "pred": prediction}

    def on_validation_batch_end(
        self, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        n_samples = 0
        bleu_1, bleu_4, meteor, rouge_l = 0, 0, 0, 0
        for pair in outputs["pred"]:
            try:
                bleu_1_, bleu_4_, meteor_, rouge_l_ = get_scores(**pair)
            except ValueError:
                bleu_1_, bleu_4_, meteor_, rouge_l_ = 0, 0, 0, 0

            bleu_1 += bleu_1_
            bleu_4 += bleu_4_
            meteor += meteor_
            rouge_l += rouge_l_

            n_samples += 1

        self.log("valid/bleu_1", bleu_1 / n_samples, on_epoch=True, prog_bar=False)
        self.log("valid/bleu_4", bleu_4 / n_samples, on_epoch=True, prog_bar=False)
        self.log("valid/meteor", meteor / n_samples, on_epoch=True, prog_bar=False)
        self.log("valid/rouge_l", rouge_l / n_samples, on_epoch=True, prog_bar=False)

    def configure_optimizers(self):
        optimizer = AdamW(
            params=self.parameters(), lr=self.lr, weight_decay=self.w_decay
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=15, eta_min=0
            ),
        }

    #########################################
    # FOR PREDICTION PURPOSE
    #########################################
    def forward(
        self,
        ques_ids,
        ques_mask,
        context_ids,
        context_mask,
    ):
        # ques_ids: [b, len_ques]
        # ques_mask: [b, len_ques]
        # context_ids: [b, n_paras, len_para]
        # context_mask: [b, n_paras, len_para]

        b = ques_mask.size()[0]

        ####################
        # Embed question, context and answer
        ####################
        ques, context = self.embd_layer.encode_ques_para(
            ques_ids=ques_ids,
            context_ids=context_ids,
            ques_mask=ques_mask,
            context_mask=context_mask,
        )
        # ques : [b, len_ques, d_hid]
        # context: [b, n_paras, len_para, d_hid]

        ####################
        # Do reasoning
        ####################
        Y = self.reasoning(ques=ques, context=context)
        # [b, len_para, d_hid]

        ####################
        # Generate answer
        ####################
        outputs = []

        generator = GeneratorOwn(
            beam_size=self.beam_size,
            init_tok=self.bert_tokenizer.cls_token_id,
            stop_tok=self.bert_tokenizer.sep_token_id,
            max_len=self.len_ans,
            model=self.generate,
            no_repeat_ngram_size=self.n_gram_beam,
            topk_strategy="topk",
        )

        for b_ in range(b):
            indices = generator.search(Y[b_, :, :])

            outputs.append(indices)

        outputs = torch.tensor(outputs, device=self.device, dtype=torch.long)

        return outputs

    def generate(self, decoder_input_ids, encoder_outputs):
        # decoder_input_ids: [list: len_]
        # encoder_outputs  : [len_para, d_hid]

        decoder_input_ids = (
            torch.LongTensor(decoder_input_ids)
            .type_as(encoder_outputs)
            .long()
            .unsqueeze(0)
        )

        decoder_input_mask = torch.ones(decoder_input_ids.shape, device=self.device)
        decoder_input_embd = self.embd_layer.encode_ans(
            input_ids=decoder_input_ids, input_masks=decoder_input_mask
        )
        # [1, len_, d_bert]

        encoder_outputs = encoder_outputs.unsqueeze(0)

        output = self.ans_infer(encoder_outputs, decoder_input_embd, decoder_input_mask)
        # [1, len_, d_vocab]

        output = output.squeeze(0)
        # [len_, d_vocab]

        return output

    def predict_step(
        self,
        batch: Any,
        batch_idx: int,
        dataloader_idx: Optional[int],
    ) -> Any:
        ques_ids = batch["ques_ids"]
        ques_mask = batch["ques_mask"]
        ans1_ids = batch["ans1_ids"]
        ans2_ids = batch["ans2_ids"]
        context_ids = batch["context_ids"]
        context_mask = batch["context_mask"]

        pred = self(
            ques_ids=ques_ids,
            ques_mask=ques_mask,
            context_ids=context_ids,
            context_mask=context_mask,
        )
        # pred: [b, len_ans, d_vocab]

        prediction = [
            {
                "pred": " ".join(self.bert_tokenizer.convert_ids_to_tokens(pred_)),
                "ref": [
                    " ".join(self.bert_tokenizer.convert_ids_to_tokens(ans1_)),
                    " ".join(self.bert_tokenizer.convert_ids_to_tokens(ans2_)),
                ],
            }
            for pred_, ans1_, ans2_ in zip(pred.squeeze(1), ans1_ids, ans2_ids)
        ]

        return prediction

    def on_predict_batch_end(
        self, outputs: Optional[Any], batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:

        #######################
        # Calculate metrics
        #######################
        n_samples = 0
        bleu_1, bleu_4, meteor, rouge_l = 0, 0, 0, 0
        for pair in outputs:
            try:
                bleu_1_, bleu_4_, meteor_, rouge_l_ = self.get_scores(**pair)
            except ValueError:
                bleu_1_, bleu_4_, meteor_, rouge_l_ = 0, 0, 0, 0

            bleu_1 += bleu_1_
            bleu_4 += bleu_4_
            meteor += meteor_
            rouge_l += rouge_l_

            n_samples += 1

        #######################
        # Log prediction and metrics
        #######################
        with open(self.path_pred, "a+") as pred_file:
            json.dump(
                {
                    "metrics": {
                        "bleu_1": bleu_1 / n_samples,
                        "bleu_4": bleu_4 / n_samples,
                        "meteor": meteor / n_samples,
                        "rouge_l": rouge_l / n_samples,
                    },
                    "predictions": outputs,
                },
                pred_file,
                indent=2,
                ensure_ascii=False,
            )
