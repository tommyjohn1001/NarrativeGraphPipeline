
from typing import Any, Optional
import json

from transformers import AdamW, BertTokenizer
from transformers.modeling_outputs import Seq2SeqLMOutput
import pytorch_lightning as plt
import torch.nn as torch_nn
import torch

from src.datamodules.narrative_datamodule import NarrativeDataModule
from src.models.layers.reasoning_layer.memorygraph_layer import GraphBasedMemoryLayer
from src.models.layers.finegrain_layer import FineGrain
from src.models.layers.ans_infer_layer import BertDecoder
from src.models.utils import CustomGen

class NarrativeModel(plt.LightningModule):
    def __init__(self,
        batch_size: int,
        seq_len_ques: int = 42,
        seq_len_para: int = 122,
        seq_len_ans: int = 42,
        n_nodes: int = 435,
        n_edges: int = 3120,
        n_gru_layers: int = 5,
        max_len_ans: int = 12,
        d_hid: int = 64,
        d_bert: int = 768,
        d_vocab: int = 32716,
        d_graph: int = 2048,
        lr: float = 5e-4,
        w_decay: float = 0.0005,
        beam_size: int = 20,
        n_gram_beam: int = 5,
        temperature: int = 1,
        topP: float = 0.5,
        path_bert: str = None,
        path_vocab: str = None,
        path_pred: str = None,
        datamodule: NarrativeDataModule = None):

        super().__init__()

        self.d_vocab        = d_vocab
        self.lr             = lr
        self.w_decay        = w_decay
        self.beam_size      = beam_size
        self.n_gram_beam    = n_gram_beam
        self.temperature    = temperature
        self.topP           = topP
        self.max_len_ans    = max_len_ans



        self.bert_tokenizer = BertTokenizer(vocab_file=path_vocab)
        self.datamodule     = datamodule

        self.path_pred = path_pred

        #############################
        # Define model
        #############################
        self.embd_layer = FineGrain(seq_len_para, n_gru_layers, d_vocab, d_bert, path_bert)
        self.reasoning  = GraphBasedMemoryLayer(batch_size, seq_len_ques, seq_len_ans, d_hid, d_bert,
                                                d_graph, n_nodes, n_edges)
        self.ans_infer  = BertDecoder(seq_len_ans, d_bert, d_vocab, path_bert)


        #############################
        # Define things
        #############################
        self.criterion  = torch_nn.CrossEntropyLoss(ignore_index=self.bert_tokenizer.pad_token_id)


    ####################################################################
    # FOR TRAINING PURPOSE
    ####################################################################

    def calc_loss(self, pred, ans1, ans2):
        # pred: [b, seq_len_ans, d_vocab]
        # ans1: [b, seq_len_ans]
        # ans2: [b, seq_len_ans]

        d_vocab     = pred.shape[2]

        pred_flat   = pred[:, :-1, :].reshape(-1, d_vocab)
        ans1_flat   = ans1[:, 1:].reshape(-1)
        ans2_flat   = ans2[:, 1:].reshape(-1)

        loss        = 0.7 * self.criterion(pred_flat, ans1_flat) +\
                      0.3 * self.criterion(pred_flat, ans2_flat)

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
        ques, paras = self.embd_layer.encode_ques_para(ques, paras, ques_mask, paras_mask)
        ans         = self.embd_layer.encode_ans(ans, ans_mask)
        # ques : [b, seq_len_ques, d_bert]
        # paras: [b, n_paras, d_bert]
        # ans  : [b, seq_len_ans, d_bert]


        ####################
        # Do reasoning
        ####################
        Y       = self.reasoning(ques, paras)
        # [b, seq_len_ans, d_bert]

        ####################
        # Generate answer
        ####################

        pred    = self.ans_infer(Y, ans, ans_mask)
        # [b, seq_len_ans, d_vocab]

        return pred


    def training_step(self, batch: Any, batch_idx: int):
        ques        = batch['ques']
        ques_mask   = batch['ques_mask']
        ans1        = batch['ans1']
        ans1_mask   = batch['ans1_mask']
        ans2        = batch['ans2']
        paras       = batch['paras']
        paras_mask  = batch['paras_mask']

        pred        = self.model(ques, ques_mask, ans1,
                                        ans1_mask, paras, paras_mask)

        loss        = self.calc_loss(pred, ans1, ans2)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=False)

        return loss

    def test_step(self, batch: Any, batch_idx: int):
        ques        = batch['ques']
        ques_mask   = batch['ques_mask']
        ans1        = batch['ans1']
        ans1_mask   = batch['ans1_mask']
        ans2        = batch['ans2']
        paras       = batch['paras']
        paras_mask  = batch['paras_mask']

        pred        = self.model(ques, ques_mask, ans1,
                                 ans1_mask, paras, paras_mask)
        # pred: [b, seq_len_ans, d_vocab]

        loss        = self.calc_loss(pred, ans1, ans2)

        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=False)

        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        ques        = batch['ques']
        ques_mask   = batch['ques_mask']
        ans1        = batch['ans1']
        ans1_mask   = batch['ans1_mask']
        ans2        = batch['ans2']
        paras       = batch['paras']
        paras_mask  = batch['paras_mask']

        pred        = self.model(ques, ques_mask, ans1,
                                 ans1_mask, paras, paras_mask)
        # pred: [b, seq_len_ans, d_vocab]

        loss        = self.calc_loss(pred, ans1, ans2)

        self.log("valid/loss", loss, on_step=False, on_epoch=True, prog_bar=False)

        return loss

    def configure_optimizers(self):
        optimizer = AdamW(params=self.parameters(), lr=self.lr,
                          weight_decay=self.w_decay)
        return {
            "optimizer"     : optimizer,
            "lr_scheduler"  : torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=0)
        }

    def on_train_batch_end(self, outputs: Optional[Any], batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        torch_nn.utils.clip_grad_value_(self.parameters(), clip_value=1)

    def on_train_epoch_end(self, unused: Optional[None]) -> None:
        if self.current_epoch % 5 == 0:
            self.datamodule.switch_answerability()

    #########################################
    # FOR PREDICTION PURPOSE
    #########################################
    def forward(self, ques, ques_mask, paras, paras_mask):
        # ques       : [b, seq_len_ques]
        # ques_mask  : [b, seq_len_ques]
        # paras      : [b, n_paras, seq_len_para]
        # paras_mask : [b, n_paras, seq_len_para]

        b   = ques.shape[0]


        ####################
        # Embed question, paras and answer
        ####################
        ques, paras = self.embd_layer.encode_ques_para(ques, paras, ques_mask, paras_mask)
        # ques : [b, seq_len_ques, d_bert]
        # paras: [b, n_paras, d_bert]


        ####################
        # Do reasoning
        ####################
        Y       = self.reasoning(ques, paras)
        # [b, seq_len_ans, d_bert]


        ####################
        # Generate answer
        ####################
        Y_  = Y.repeat_interleave(self.beam_size, dim=0)
        # [b_, seq_len_ans, d_bert]

        generator  = CustomGen(
            batch_size=b,
            max_length=self.max_len_ans,
            num_beams=self.beam_size,
            temperature=self.temperature,
            no_repeat_ngram_size=self.n_gram_beam,
            model=self.generate,
            pad_token_id=self.bert_tokenizer.pad_token_id,
            bos_token_id=self.bert_tokenizer.cls_token_id,
            eos_token_id=self.bert_tokenizer.sep_token_id)

        outputs = generator.beam_sample(None, Y_)

        return outputs

    def generate(self, decoder_input_ids, encoder_outputs):
        # decoder_input_ids: [b_, seq_len<=200]
        # encoder_outputs  : [b_, seq_len_, d_bert]

        b_, seq_len = decoder_input_ids.shape

        decoder_input_mask  = torch.ones((b_, seq_len))
        decoder_input_embd  = self.embd_layer.encode_ans(decoder_input_ids,
                                                         decoder_input_mask)
        # [b_, seq=*, d_bert]

        output  = self.ans_infer(encoder_outputs, decoder_input_embd, decoder_input_mask)
        # [b_, seq=*, d_vocab]

        return Seq2SeqLMOutput(logits=output)


    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int]) -> Any:
        ques        = batch['ques']
        ques_mask   = batch['ques_mask']
        ans1        = batch['ans1']
        ans1_mask   = batch['ans1_mask']
        ans2        = batch['ans2']
        paras       = batch['paras']
        paras_mask  = batch['paras_mask']


        pred        = self(ques, ques_mask, ans1,
                           ans1_mask, paras, paras_mask)

        prediction  = [
            {
                "pred"  : ' '.join(self.bert_tokenizer.convert_ids_to_tokens(pred_)),
                "ans1"  : ' '.join(self.bert_tokenizer.convert_ids_to_tokens(ans1_)),
                "ans2"  : ' '.join(self.bert_tokenizer.convert_ids_to_tokens(ans2_))
            } for pred_, ans1_, ans2_ in zip(pred, ans1, ans2)
        ]

        return prediction

    def on_predict_batch_end(self, outputs: Optional[Any], batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        with open(self.path_pred, 'a+') as pred_file:
            json.dump(outputs, pred_file, indent=2, ensure_ascii=False)


    #########################################
    # OTHER METHODS
    #########################################
    def get_memory(self, memory: torch.Tensor):
        pass

    def load_memory(self):
        pass
