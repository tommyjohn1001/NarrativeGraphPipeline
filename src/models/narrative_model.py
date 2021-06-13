from typing import Any, Optional
import json

from transformers import AdamW, BertTokenizer
from transformers.modeling_outputs import Seq2SeqLMOutput
import pytorch_lightning as plt
import torch.nn.functional as torch_F
import torch.nn as torch_nn
import torch

from src.datamodules.narrative_datamodule import NarrativeDataModule
from src.models.layers.reasoning_layer.memory_layer import MemoryBasedReasoning
from src.models.layers.bertbasedembd_layer import BertBasedLayer
from src.models.layers.ans_infer_layer import BertDecoder
from src.utils.utils import CustomGen, get_scores


class NarrativeModel(plt.LightningModule):
    def __init__(
        self,
        seq_len_ques: int = 42,
        seq_len_para: int = 162,
        seq_len_ans: int = 42,
        max_len_ans: int = 12,
        n_paras: int = 30,
        n_layers_gru: int = 5,
        n_layers_trans: int = 3,
        n_heads_trans: int = 4,
        d_hid: int = 64,
        d_bert: int = 768,
        d_vocab: int = 30522,
        lr: float = 1e-5,
        w_decay: float = 1e-2,
        beam_size: int = 20,
        n_gram_beam: int = 5,
        temperature: int = 1,
        topP: float = 0.5,
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
        self.temperature = temperature
        self.topP = topP
        self.max_len_ans = max_len_ans

        self.bert_tokenizer = BertTokenizer.from_pretrained(path_bert)
        self.datamodule = datamodule

        self.path_pred = path_pred
        self.path_train_pred = path_train_pred

        #############################
        # Define model
        #############################
        self.embd_layer = BertBasedLayer(d_bert, path_bert)
        self.reasoning = MemoryBasedReasoning(
            seq_len_ques,
            seq_len_para,
            seq_len_ans,
            n_paras,
            n_layers_gru,
            n_heads_trans,
            n_layers_trans,
            d_hid,
            d_bert,
            self.device,
        )
        self.ans_infer = BertDecoder(seq_len_ans, d_bert, d_vocab)

        #############################
        # Define things
        #############################
        self.criterion = torch_nn.CrossEntropyLoss(
            ignore_index=self.bert_tokenizer.pad_token_id
        )

    ####################################################################
    # FOR TRAINING PURPOSE
    ####################################################################

    def calc_loss(self, pred, ans):
        # pred: [b, seq_len_ans, d_vocab]
        # ans: [b, seq_len_ans]

        d_vocab = pred.shape[2]

        pred_flat = pred[:, :-1, :].reshape(-1, d_vocab)
        ans_flat = ans[:, 1:].reshape(-1)

        loss = self.criterion(pred_flat, ans_flat)

        return loss

    def model(self, ques, ques_mask, ans, ans_mask, paras, paras_mask):
        # ques       : [b, seq_len_ques]
        # ques_mask  : [b, seq_len_ques]
        # paras      : [b, n_paras, seq_len_para]
        # paras_mask : [b, n_paras, seq_len_para]
        # ans        : [b, seq_len_ans]
        # ans_mask   : [b, seq_len_ans]

        ####################
        # Embed question, paras and answer
        ####################
        ques, paras = self.embd_layer.encode_ques_para(
            ques, paras, ques_mask, paras_mask
        )
        ans = self.embd_layer.encode_ans(ans, ans_mask)
        # ques : [b, seq_len_ques, d_bert]
        # paras: [b, n_paras, seq_len_para, d_bert]
        # ans  : [b, seq_len_ans, d_bert]

        ####################
        # Do reasoning
        ####################
        Y = self.reasoning(ques, paras, paras_mask)
        # [b, seq_len_ans, d_bert]

        ####################
        # Generate answer
        ####################
        pred = self.ans_infer(Y, ans, ans_mask)
        # pred: [b, seq_len_ans, d_vocab]

        return pred

    def training_step(self, batch: Any, batch_idx: int):
        ques = batch["ques"]
        ques_mask = batch["ques_mask"]
        ans1 = batch["ans1"]
        ans2 = batch["ans2"]
        ans1_mask = batch["ans1_mask"]
        paras = batch["paras"]
        paras_mask = batch["paras_mask"]

        pred = self.model(ques, ques_mask, ans1, ans1_mask, paras, paras_mask)

        loss = self.calc_loss(pred, ans1)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=False)

        _, prediction = torch.topk(torch_F.log_softmax(pred, dim=2), 1, dim=2)

        prediction = [
            {
                "pred": " ".join(self.bert_tokenizer.convert_ids_to_tokens(pred_)),
                "ans1": " ".join(self.bert_tokenizer.convert_ids_to_tokens(ans1_)),
                "ans2": " ".join(self.bert_tokenizer.convert_ids_to_tokens(ans2_)),
            }
            for pred_, ans1_, ans2_ in zip(prediction, ans1, ans2)
        ]

        return {"loss": loss, "pred": prediction}

    def on_train_batch_end(
        self, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        with open(self.path_train_pred, "a+") as pred_file:
            json.dump(outputs["pred"], pred_file, indent=2, ensure_ascii=False)

    def test_step(self, batch: Any, batch_idx: int):
        ques = batch["ques"]
        ques_mask = batch["ques_mask"]
        ans1 = batch["ans1"]
        ans1_mask = batch["ans1_mask"]
        paras = batch["paras"]
        paras_mask = batch["paras_mask"]

        pred = self.model(ques, ques_mask, ans1, ans1_mask, paras, paras_mask)

        loss = self.calc_loss(pred, ans1)

        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=False)

        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        ques = batch["ques"]
        ques_mask = batch["ques_mask"]
        ans1 = batch["ans1"]
        ans2 = batch["ans2"]
        paras = batch["paras"]
        paras_mask = batch["paras_mask"]

        pred = self(ques, ques_mask, paras, paras_mask)

        _, prediction = torch.topk(torch_F.log_softmax(pred, dim=2), 1, dim=2)

        prediction = [
            {
                "pred": " ".join(self.bert_tokenizer.convert_ids_to_tokens(pred_)),
                "ref": [
                    " ".join(self.bert_tokenizer.convert_ids_to_tokens(ans1_)),
                    " ".join(self.bert_tokenizer.convert_ids_to_tokens(ans2_)),
                ],
            }
            for pred_, ans1_, ans2_ in zip(prediction, ans1, ans2)
        ]

        return prediction

    def on_validation_batch_end(
        self, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        n_samples = 0
        bleu_1, bleu_4, meteor, rouge_l = 0, 0, 0, 0
        for pair in outputs:
            bleu_1_, bleu_4_, meteor_, rouge_l_ = get_scores(**pair)

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
    def forward(self, ques, ques_mask, paras, paras_mask):
        # ques       : [b, seq_len_ques]
        # ques_mask  : [b, seq_len_ques]
        # paras      : [b, n_paras, seq_len_para]
        # paras_mask : [b, n_paras, seq_len_para]

        b = ques.shape[0]

        ####################
        # Embed question, paras and answer
        ####################
        ques, paras = self.embd_layer.encode_ques_para(
            ques, paras, ques_mask, paras_mask
        )
        # ques : [b, seq_len_ques, d_bert]
        # paras: [b, n_paras, d_bert]

        ####################
        # Do reasoning
        ####################
        Y = self.reasoning(ques, paras)
        # [b, seq_len_ans, d_bert]

        ####################
        # Generate answer
        ####################
        Y_ = Y.repeat_interleave(self.beam_size, dim=0)
        # [b_, seq_len_ans, d_bert]

        generator = CustomGen(
            batch_size=b,
            max_length=self.max_len_ans,
            num_beams=self.beam_size,
            temperature=self.temperature,
            no_repeat_ngram_size=self.n_gram_beam,
            model=self.generate,
            pad_token_id=self.bert_tokenizer.pad_token_id,
            bos_token_id=self.bert_tokenizer.cls_token_id,
            eos_token_id=self.bert_tokenizer.sep_token_id,
        )

        outputs = generator.beam_sample(None, Y_)

        return outputs

    def generate(self, decoder_input_ids, encoder_outputs):
        # decoder_input_ids: [b_, seq_len<=200]
        # encoder_outputs  : [b_, seq_len_, d_bert]

        b_, seq_len = decoder_input_ids.shape

        decoder_input_mask = torch.ones((b_, seq_len))
        decoder_input_embd = self.embd_layer.encode_ans(
            decoder_input_ids, decoder_input_mask
        )
        # [b_, seq=*, d_bert]

        output = self.ans_infer(encoder_outputs, decoder_input_embd, decoder_input_mask)
        # [b_, seq=*, d_vocab]

        return Seq2SeqLMOutput(logits=output)

    def predict_step(
        self,
        batch: Any,
        batch_idx: int,
        dataloader_idx: Optional[int],
    ) -> Any:
        ques = batch["ques"]
        ques_mask = batch["ques_mask"]
        ans1 = batch["ans1"]
        ans2 = batch["ans2"]
        ans_mask = batch["ans2_mask"]
        paras = batch["paras"]
        paras_mask = batch["paras_mask"]

        pred = self.model(ques, ques_mask, ans2, ans2_mask, paras, paras_mask)

        prediction = [
            {
                "pred": " ".join(self.bert_tokenizer.convert_ids_to_tokens(pred_)),
                "ans1": " ".join(self.bert_tokenizer.convert_ids_to_tokens(ans1_)),
                "ans2": " ".join(self.bert_tokenizer.convert_ids_to_tokens(ans2_)),
            }
            for pred_, ans1_, ans2_ in zip(pred, ans1, ans2)
        ]

        return prediction

    def on_predict_batch_end(
        self,
        outputs: Optional[Any],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:

        n_samples = 0
        metrics = {
            "bleu_1": 0,
            "bleu_4": 0,
            "meteor": 0,
            "rouge_l": 0,
        }

        for pair in outputs:
            bleu_1_, bleu_4_, meteor_, rouge_l_ = get_scores(**pair)

            metrics["bleu_1"] += bleu_1_
            metrics["bleu_4"] += bleu_4_
            metrics["meteor"] += meteor_
            metrics["rouge_l"] += rouge_l_

            n_samples += 1

        metrics["bleu_1"] /= n_samples
        metrics["bleu_4"] /= n_samples
        metrics["meteor"] /= n_samples
        metrics["rouge_l"] /= n_samples

        with open(self.path_pred, "a+") as pred_file:
            json.dump(metrics, pred_file, indent=2, ensure_ascii=False)
