import os

from torch.utils.data import DataLoader
import torch.nn.functional as torch_f
import torch.nn as torch_nn
import torch
from transformers import AdamW


from modules.narrativepipeline.utils_origin import TrainDataset, EvalDataset, build_vocab_PGD, Vocab
from modules.pg_decoder.PointerGeneratorDecoder import PointerGeneratorDecoder
from modules.utils import EmbeddingLayer, check_file_existence, transpose, get_scores
from modules.Reasoning.IAL import IntrospectiveAlignmentLayer
from configs import args, logging, PATH

class  NarrativePipeline(torch_nn.Module):
    def __init__(self):
        super().__init__()

        self.embd_layer = EmbeddingLayer()
        self.reasoning  = IntrospectiveAlignmentLayer()
        self.pg_decoder = PointerGeneratorDecoder()

    def forward(self, ques, contx, ans, ans_mask):
        # ques      : [batch, seq_len_ques, d_embd]
        # contx     : [batch, seq_len_contx, d_embd]
        # ans       : [batch, max_len_ans, d_embd]
        # ans_mask  : [batch, max_len_ans]
        # NOTE: all arguments used in this method must not modify the orginal

        ####################
        # Embed question and context
        # by LSTM
        ####################
        H_q = self.embd_layer(ques)
        H_c = self.embd_layer(contx)
        # H_q   : [batch, seq_len_ques, d_hid]
        # H_c   : [batch, seq_len_contex, d_hid]


        ####################
        # Do reasoning with IAL
        ####################
        Y       = self.reasoning(H_q, H_c)
        # Y: [batch, seq_len_context, 2*d_hid]


        ####################
        # Generate answer with PGD
        ####################
        pred    = self.pg_decoder(Y, H_q, ans, ans_mask)
        # pred: [batch, max_len_ans, d_vocab + seq_len_cntx]

        return pred


class Trainer():
    def __init__(self):
        super().__init__()

        self.vocab  = Vocab(PATH['vocab_PGD'])
        ################################
        # Build vocab for PointerGeneratorDecoder
        ################################
        logging.info("Preparation: Build vocab for PGD")
        if not check_file_existence(PATH['vocab_PGD']):
            build_vocab_PGD()

    def save_model(self, model):
        """
        Save model during training
        :param model: Pytorch model
        """
        torch.save(model.state_dict(), PATH['saved_model'])

    def load_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Load model from path

        Args:
            model (object): model to be load from saved instance if existed

        Returns:
            model even loaded or not
        """

        if check_file_existence(PATH['saved_model']):
            logging.info("=> Saved model instance existed. Load it.")
            model.load_state_dict(torch.load(PATH['saved_model']), strict=False)

        return model

    def train(self, model, dataset_train, criterion, optimizer):
        n_samples       = len(dataset_train)

        model.train()

        iterator_train  = DataLoader(dataset_train, batch_size=args.batch, shuffle=True)
        nth_batch   = 0
        loss_train  = 0
        for batch in iterator_train:
            ques        = batch['ques']
            contx       = batch['contx']
            # NOTE: Due to limitation, this only uses answer1 only. Later, both answers must be made used of.
            ans         = batch['ans1']
            ans_mask    = batch['ans1_mask']
            ans_tok_idx = batch['ans1_tok_idx']

            optimizer.zero_grad()

            pred        = model(ques, contx, ans, ans_mask)
            # pred: [batch, max_len_ans, d_vocab + seq_len_cntx]
            pred        = transpose(pred)
            # pred: [batch, d_vocab + seq_len_cntx, max_len_ans]
            loss        = criterion(pred, ans_tok_idx)

            loss.backward()
            torch_nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
            optimizer.step()

            loss_train += loss

            logging.info(f"  train: batch {nth_batch} | loss: {loss:5f}")
            nth_batch += 1

        return loss_train / n_samples


    def test(self, model, dataset_test, criterion):
        n_samples       = len(dataset_test)

        model.eval()

        iterator_test   = DataLoader(dataset_test, batch_size=args.batch)
        nth_batch       = 0
        loss_test       = 0
        with torch.no_grad():
            for batch in iterator_test:
                ques        = batch['ques']
                contx       = batch['contx']
                # NOTE: Due to limitation, this only uses answer1 only. Later, both answers must be made used of.
                ans         = batch['ans1']
                ans_mask    = batch['ans1_mask']
                ans_tok_idx = batch['ans1_tok_idx']

                pred        = model(ques, contx, ans, ans_mask)
                # pred: [batch, max_len_ans, d_vocab + seq_len_cntx]
                pred        = transpose(pred)
                # pred: [batch, d_vocab + seq_len_cntx, max_len_ans]
                loss        = criterion(pred, ans_tok_idx)

                loss_test += loss

                logging.info(f"  test: batch {nth_batch} | loss: {loss:5f}")
                nth_batch += 1

        return loss_test / n_samples


    def trigger_train(self):
        ###############################
        # Load data
        ###############################
        dataset_train   = TrainDataset(os.path.dirname(PATH['dataset_para']).replace("[SPLIT]", "train"),
                                       PATH['vocab_PGD'])
        dataset_test    = EvalDataset(os.path.dirname(PATH['dataset_para']).replace("[SPLIT]", "test"),
                                      PATH['vocab_PGD'])


        ###############################
        # Defind model and associated stuffs
        ###############################
        model       = NarrativePipeline().to(args.device)
        optimizer   = AdamW(model.parameters(), lr=args.lr, weight_decay=args.w_decay)
        criterion   = torch_nn.CrossEntropyLoss()


        ###############################
        # Start training
        ###############################
        best_loss_test  = 10e10
        for nth_epoch in range(args.n_epochs):
            logging.info(f"= Epoch {nth_epoch}")

            loss_train  = self.train(model, dataset_train, criterion, optimizer)
            loss_test   = self.test(model, dataset_test, criterion)

            if loss_test > best_loss_test:
                if dataset_train.n_exchange < args.n_paras:
                    dataset_train.switch_answerability()                   
                else:
                    # NOTE: Later, at this position switch_understandability() must be called
                    break
            else:
                best_loss_test = loss_test
                self.save_model(model)

            logging.info(f"= Epoch {nth_epoch} finishes: loss_train {loss_train:.5f} | loss_test {loss_test:.5f}")


    def get_batch_scores(self, pred, ans_tok_idx):
        """Calculate BLEU-1, BLEU4, METEOR and ROUGE_L for each entry in batch.

        Args:
            pred (tensor): predicted tensor
            ans_tok_idx (tensor): target answer
        """
        # pred: [batch, d_vocab + seq_len_cntx, max_len_ans]
        # ans_tok_idx: [batch, max_len_ans]
        batch   = pred.shape[0]

        pred_   = torch.argmax(torch_f.log_softmax(pred, 1), 1)
        # pred_: [batch, max_len_ans]

        bleu_1, bleu_4, meteor, rouge_l = 0, 0, 0, 0

        for p, a in zip(pred_.tolist(), ans_tok_idx.tolist()):
            p_   = ' '.join([self.vocab.itos[i - 1] for i in p if i > 0])
            a_    = ' '.join([self.vocab.itos[i - 1] for i in a if i > 0])

            bleu_1_, bleu_4_, meteor_, rouge_l_ = get_scores(p_, a_)

            bleu_1 += bleu_1_
            bleu_4 += bleu_4_
            meteor += meteor_
            rouge_l += rouge_l_

        return bleu_1, bleu_4, meteor, rouge_l


    def trigger_infer(self):
        # NOTE: Use scorer in this method
        ###############################
        # Load data
        ###############################
        dataset_valid   = EvalDataset(os.path.dirname(PATH['dataset_para']).replace("[SPLIT]", "validation"),
                                      PATH['vocab_PGD'])

        ###############################
        # Defind model and associated stuffs
        ###############################
        model       = NarrativePipeline().to(args.device)
        criterion   = torch_nn.CrossEntropyLoss()
        model = model.to(args.device)

        model       = self.load_model(model)

        ###############################
        # Start infering
        ###############################
        n_samples       = len(dataset_valid)
        bleu_1, bleu_4, meteor, rouge_l = 0, 0, 0, 0

        model.eval()
        iterator_valid  = DataLoader(dataset_valid, batch_size=args.batch)

        with torch.no_grad():
            nth_batch   = 0
            loss_test   = 0
            for batch in iterator_valid:
                ques        = batch['ques']
                contx       = batch['contx']
                # NOTE: Due to limitation, this only uses answer1 only. Later, both answers must be made used of.
                ans         = batch['ans1']
                ans_mask    = batch['ans1_mask']
                ans_tok_idx = batch['ans1_tok_idx']

                pred        = model(ques, contx, ans, ans_mask)
                # pred: [batch, max_len_ans, d_vocab + seq_len_cntx]
                pred        = transpose(pred)
                # pred: [batch, d_vocab + seq_len_cntx, max_len_ans]
                loss        = criterion(pred, ans_tok_idx)

                # Calculate loss
                loss_test += loss

                # Get batch score
                bleu_1_, bleu_4_, meteor_, rouge_l_ = self.get_batch_scores(pred, ans_tok_idx)

                bleu_1 += bleu_1_
                bleu_4 += bleu_4_
                meteor += meteor_
                rouge_l += rouge_l_

                logging.info(f"  validate: batch {nth_batch} | loss: {loss:5f}")
                nth_batch += 1


        logging.info(f"End validattion: bleu_1 {bleu_1/n_samples:.5f} | bleu_4 {bleu_4/n_samples:.5f} \
                     | meteor {meteor/n_samples:.5f} | rouge_l {rouge_l/n_samples:.5f}")

if __name__ == '__main__':
    logging.info("* Start NarrativePipeline")


    narrative_pipeline  = Trainer()

    narrative_pipeline.trigger_train()

    narrative_pipeline.trigger_infer()
