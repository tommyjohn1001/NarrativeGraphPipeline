from typing import Any, Optional
import json, re


from transformers import AdamW, BertTokenizer
from transformers.modeling_outputs import Seq2SeqLMOutput
import pytorch_lightning as plt
import torch.nn.functional as torch_F
import torch.nn as torch_nn
import torch
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
import numpy as np

from src.datamodules.narrative_datamodule import NarrativeDataModule
from src.models.layers.reasoning_layer.memorygraph_layer import GraphBasedMemoryLayer
from src.models.layers.finegrain_layer import FineGrain
from src.models.layers.ans_infer_layer import BertDecoder
from src.models.utils import BeamSearchHuggingface, BeamSearchOwn

EPSILON = 10e-10


class NarrativeModel(plt.LightningModule):
    def __init__(
        self,
        batch_size: int,
        seq_len_ques: int = 42,
        seq_len_para: int = 170,
        seq_len_ans: int = 15,
        n_nodes: int = 10,
        n_edges: int = 40,
        n_gru_layers: int = 5,
        max_len_ans: int = 12,
        d_hid: int = 64,
        d_bert: int = 768,
        d_vocab: int = 30522,
        d_graph: int = 2048,
        lr: float = 1e-5,
        w_decay: float = 1e-2,
        switch_frequency: int = 5,
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
        self.switch_frequency = switch_frequency

        self.bert_tokenizer = BertTokenizer.from_pretrained(path_bert)
        self.datamodule = datamodule

        self.path_pred = path_pred
        self.path_train_pred = path_train_pred

        #############################
        # Define model
        #############################
        self.embd_layer = FineGrain(
            seq_len_para,
            n_gru_layers,
            d_bert,
            path_bert,
        )
        self.reasoning = GraphBasedMemoryLayer(
            batch_size,
            seq_len_ques,
            seq_len_ans,
            d_hid,
            d_bert,
            d_graph,
            n_nodes,
            n_edges,
        )
        self.ans_infer = BertDecoder(
            seq_len_ans,
            d_bert,
            d_vocab,
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

    def process_sent(self, sent: str):
        return re.sub(r"(\[PAD\]|\[CLS\]|\[SEP\]|\[UNK\]|\[MASK\])", "", sent).strip()

    def get_scores(self, ref: list, pred: str):
        """Calculate metrics BLEU-1, BLEU4, METEOR and ROUGE_L.

        ref = [
            "the transcript is a written version of each day",
            "version of each day"
        ]
        pred= "a written version of"

        Args:
            ref (list): list of reference strings
            pred (str): string generated by model

        Returns:
            tuple: tuple of 4 scores



        """
        pred = self.process_sent(pred)
        ref = list(map(self.process_sent, ref))

        # Calculate BLEU score
        ref_ = [x.split() for x in ref]
        pred_ = pred.split()

        bleu_1 = sentence_bleu(ref_, pred_, weights=(1, 0, 0, 0))
        bleu_4 = sentence_bleu(ref_, pred_, weights=(0.25, 0.25, 0.25, 0.25))

        # Calculate METEOR
        meteor = meteor_score(ref, pred)

        # Calculate ROUGE-L
        scores = np.array(
            [Rouge().get_scores(ref_, pred, avg=True)["rouge-l"]["f"] for ref_ in ref]
        )
        rouge_l = np.mean(scores)

        return (
            bleu_1 if bleu_1 > EPSILON else 0,
            bleu_4 if bleu_4 > EPSILON else 0,
            meteor if meteor > EPSILON else 0,
            rouge_l if rouge_l > EPSILON else 0,
        )

    def model(
        self,
        ques_ids,
        ques_mask,
        ans_ids,
        ans_mask,
        context_ids,
        context_mask,
    ):
        # ques       : [b, seq_len_ques]
        # ques_mask  : [b, seq_len_ques]
        # context_ids  : [b, n_paras, seq_len_para]
        # context_mask : [b, n_paras, seq_len_para]
        # ans        : [b, seq_len_ans]
        # ans_mask   : [b, seq_len_ans]

        ####################
        # Embed question, context and answer
        ####################
        ques, context = self.embd_layer.encode_ques_para(
            ques_ids=ques_ids,
            context_ids=context_ids,
            ques_mask=ques_mask,
            context_mask=context_mask,
        )
        ans = self.embd_layer.encode_ans(ans_ids=ans_ids, ans_mask=ans_mask)
        # ques : [b, seq_len_ques, d_bert]
        # context: [b, n_paras, d_bert]
        # ans  : [b, seq_len_ans, d_bert]

        ####################
        # Do reasoning
        ####################
        Y = self.reasoning(ques, context)
        # [b, seq_len_ans, d_bert]

        ####################
        # Generate answer
        ####################

        pred = self.ans_infer(Y, ans, ans_mask)
        # [b, seq_len_ans, d_vocab]

        return pred

    def training_step(self, batch: Any, batch_idx: int):
        ques_ids = batch["ques_ids"]
        ques_mask = batch["ques_mask"]
        ans1_ids = batch["ans1_ids"]
        ans2_ids = batch["ans2_ids"]
        ans1_mask = batch["ans1_mask"]
        context_ids = batch["context_ids"]
        context_mask = batch["context_mask"]

        pred = self.model(
            ques_ids, ques_mask, ans1_ids, ans1_mask, context_ids, context_mask
        )
        # [b, d_vocab, seq_len_ans]

        loss = self.criterion(pred, ans1_ids)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=False)

        _, prediction = torch.topk(torch_F.log_softmax(pred, dim=1), 1, dim=1)

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

        return {"loss": loss, "pred": prediction}

    def on_train_batch_end(
        self, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        with open(self.path_train_pred, "a+") as pred_file:
            json.dump(outputs["pred"], pred_file, indent=2, ensure_ascii=False)

    def test_step(self, batch: Any, batch_idx: int):
        ques_ids = batch["ques_ids"]
        ques_mask = batch["ques_mask"]
        ans1_ids = batch["ans1_ids"]
        ans1_mask = batch["ans1_mask"]
        context_ids = batch["context_ids"]
        context_mask = batch["context_mask"]

        pred = self.model(
            ques_ids, ques_mask, ans1_ids, ans1_mask, context_ids, context_mask
        )

        loss = self.criterion(pred, ans1_ids)

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

        pred = self.model(
            ques_ids, ques_mask, ans1_ids, ans1_mask, context_ids, context_mask
        )
        # [b, d_vocab, seq_len_ans]

        loss = self.criterion(pred, ans1_ids)

        self.log("valid/loss", loss, on_step=False, on_epoch=True, prog_bar=False)

        _, prediction = torch.topk(torch_F.log_softmax(pred, dim=1), 1, dim=1)

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

        return {"loss": loss, "pred": prediction}

    def on_validation_batch_end(
        self, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        n_samples = 0
        bleu_1, bleu_4, meteor, rouge_l = 0, 0, 0, 0
        for pair in outputs["pred"]:
            bleu_1_, bleu_4_, meteor_, rouge_l_ = self.get_scores(**pair)

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

    def on_train_epoch_end(self) -> None:
        if self.current_epoch % self.switch_frequency == 0 and self.current_epoch != 0:
            self.datamodule.switch_answerability()

    #########################################
    # FOR PREDICTION PURPOSE
    #########################################
    def forward(self, ques_ids, ques_mask, context_ids, context_mask):
        # ques_ids       : [b, seq_len_ques]
        # ques_mask  : [b, seq_len_ques]
        # context_ids      : [b, n_paras, seq_len_para]
        # paras_mask : [b, n_paras, seq_len_para]

        b = ques_ids.size(0)

        ####################
        # Embed question, context
        ####################
        ques, context = self.embd_layer.encode_ques_para(
            ques_ids=ques_ids,
            context_ids=context_ids,
            ques_mask=ques_mask,
            context_mask=context_mask,
        )
        # ques : [b, seq_len_ques, d_bert]
        # context: [b, n_paras, d_bert]

        ####################
        # Do reasoning
        ####################
        Y = self.reasoning(ques, context)
        # [b, seq_len_ans, d_bert]

        ####################
        # Generate answer
        ####################
        # NOTE: This belongs to BeamSearchHugging and therefore is commented
        # Y_ = Y.repeat_interleave(self.beam_size, dim=0)
        # # [b_, seq_len_ans, d_bert]

        # generator = BeamSearchHuggingface(
        #     batch_size=b,
        #     max_length=self.max_len_ans,
        #     num_beams=self.beam_size,
        #     temperature=self.temperature,
        #     no_repeat_ngram_size=self.n_gram_beam,
        #     model=self.generate,
        #     pad_token_id=self.bert_tokenizer.pad_token_id,
        #     bos_token_id=self.bert_tokenizer.cls_token_id,
        #     eos_token_id=self.bert_tokenizer.sep_token_id,
        # )

        # outputs = generator.beam_sample(None, Y_)

        outputs = []

        beam_search = BeamSearchOwn(
            beam_size=self.beam_size,
            init_tok=self.bert_tokenizer.cls_token_id,
            stop_tok=self.bert_tokenizer.sep_token_id,
            max_len=self.max_len_ans,
            model=self.generate_own,
            no_repeat_ngram_size=self.n_gram_beam,
            topk_strategy="select_mix_beam",
        )

        for b_ in range(b):
            indices = beam_search.search(Y[b_, :, :])

            outputs.append(indices)

        outputs = torch.tensor(outputs, device=self.device, dtype=torch.long)

        return outputs

    # NOTE: This belongs to BeamSearchHugging and therefore is commented
    # def generate(self, decoder_input_ids, encoder_outputs):
    #     # decoder_input_ids: [b_, seq_len<=200]
    #     # encoder_outputs  : [b_, seq_len_, d_bert]

    #     b_, seq_len = decoder_input_ids.shape

    #     decoder_input_mask = torch.ones((b_, seq_len))
    #     decoder_input_embd = self.embd_layer.encode_ans(
    #         decoder_input_ids, decoder_input_mask
    #     )
    #     # [b_, seq=*, d_bert]

    #     output = self.ans_infer(encoder_outputs, decoder_input_embd, decoder_input_mask)
    #     # [b_, seq=*, d_vocab]

    #     return Seq2SeqLMOutput(logits=output)

    def generate_own(self, decoder_input_ids, encoder_outputs):
        # decoder_input_ids: [seq_len<=200]
        # encoder_outputs  : [seq_len_ans, d_bert]

        decoder_input_ids = (
            torch.LongTensor(decoder_input_ids)
            .type_as(encoder_outputs)
            .long()
            .unsqueeze(0)
        )

        decoder_input_mask = torch.ones(decoder_input_ids.shape, device=self.device)
        decoder_input_embd = self.embd_layer.encode_ans(
            decoder_input_ids, decoder_input_mask
        )
        # [1, seq=*, d_bert]

        encoder_outputs = encoder_outputs.unsqueeze(0)

        output = self.ans_infer(encoder_outputs, decoder_input_embd, decoder_input_mask)
        # [1, seq=*, d_vocab]

        output = output.squeeze(0)
        # [seq=*, d_vocab]

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

        prediction = self(
            ques_ids=ques_ids,
            ques_mask=ques_mask,
            context_ids=context_ids,
            context_mask=context_mask,
        )

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

    def on_predict_batch_end(
        self, outputs: Optional[Any], batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        with open(self.path_pred, "a+") as pred_file:
            json.dump(outputs, pred_file, indent=2, ensure_ascii=False)
