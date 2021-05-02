import os, json

from torch.utils.data import DataLoader
import torch.nn.functional as torch_f
import torch.nn as torch_nn
import torch
from transformers import AdamW


from modules.narrativepipeline.utils import CustomDataset, build_vocab, Vocab
# from modules.ans_infer.PointerGeneratorDecoder import PointerGeneratorDecoder
from modules.ans_infer.Transformer import TransDecoder
from modules.utils import check_exist, get_scores
from modules.Reasoning.GraphReasoning import GraphReasoning
from modules.finegrained.BertEmbedding import SimpleBertEmbd
from configs import args, logging, PATH

class  NarrativePipeline(torch_nn.Module):
    def __init__(self, vocab):
        super().__init__()

        self.embd_layer = SimpleBertEmbd()
        self.reasoning  = GraphReasoning()
        self.ans_infer  = TransDecoder(vocab)

    def forward(self, ques, ques_mask, ans, ans_mask,
                paras, paras_len, paras_mask, edge_indx,
                edge_len, is_inferring=False):
        # ques      : [b, seq_len_ques]
        # ques_mask : [b, seq_len_ques]
        # paras     : [b, n_paras, seq_len_para]
        # paras_len : [b]
        # paras_mask: [b, n_paras, seq_len_para]
        # ans       : [b, seq_len_ans]
        # ans_mask  : [b, seq_len_ans]
        # edge_indx : [b, 2, n_edges]
        # edge_len  : [b]

        batch   = ques.shape[0]

        ####################
        # Embed question, paras and answer
        # with BERT
        ####################
        ques_embd, ques_seq_embd, para_embd, ans_seq_embd = \
            self.embd_layer(ques, ques_mask, paras, paras_mask, ans, ans_mask)
        # ques_embd     : [b, 768]
        # ques_seq_embd : [b, seq_len_ques, 768]
        # paras_seq_embd: [b, n_paras, 768]
        # ans_seq_embd  : [b, 768]

        ####################
        # Do reasoning with IAL
        ####################

        # Add question to paras to create tensor 'node_feat'
        node_feat   = torch.cat((ques_embd.unsqueeze(1), para_embd), dim=1).to(args.device)
        # node_feat : [b, 1+n_paras, 768]

        # Add 1 to each of tensor 'paras_len' because question is added to paras
        paras_len   = paras_len + torch.ones((batch,))

        Y       = self.reasoning(node_feat, edge_indx, paras_len.int(), edge_len)
        # Y: [b, d_hid=256]

        ####################
        # Generate answer with PGD
        ####################
        # Repeat tensor 'Y' to dim [b, seq_len_para * n_paras, d_hid*2]
        d       = args.seq_len_para * args.n_paras
        Y       = Y.unsqueeze(1).repeat(1, d, 2)
        # [b, seq_len_para * n_paras, d_hid*2]

        pred    = self.ans_infer(Y, ques_seq_embd, ans_seq_embd,
                                 ans_mask, is_inferring)
        # pred: [batch, max_len_ans, d_vocab + seq_len_cntx]

        return pred


class Trainer():
    def __init__(self):
        super().__init__()

        ################################
        # Build vocab for PointerGeneratorDecoder
        ################################
        logging.info("Preparation: Build vocab for PGD")
        if not check_exist(PATH['vocab']):
            build_vocab()


        self.vocab  = Vocab(PATH['vocab'])

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
        model.load_state_dict(torch.load(PATH['saved_model']), strict=False)

        return model

    def save_checkpoint(self, model, optimizer, scheduler, epoch, best_loss_test):
        """Save training checkpoint.

        Args:
            model (torch.nn.Module): model to be saved
            optimizer (torch.nn.optim): optimizer
            epoch (int): no. trained epochs
            loss (tensor): loss
        """
        torch.save(
            {
               'epoch'          : epoch,
               'model_state'    : model.state_dict(),
               'optim_state'    : optimizer.state_dict(),
               'sched_state'    : scheduler.state_dict(),
               'best_loss_test' : best_loss_test
            }, PATH['saved_chkpoint'])

    def load_checkpoint(self):
        """Load state of model and optimizer for continuing training.

        Returns:
            dict: dict containing necessary info for training
        """
        if check_exist(PATH['saved_chkpoint']):
            logging.info("=> Saved checkpoint instance existed. Load it.")
        return torch.load(PATH['saved_chkpoint'])

    def train(self, model, dataset_train:CustomDataset, criterion, optimizer, scheduler):
        loss_train  = 0
        nth_batch   = 0

        model.train()

        for train_file in dataset_train.file_names:
            logging.info(f"Train with file: {train_file}")

            dataset_train.read_shard(train_file)
            iterator_train  = DataLoader(dataset_train, batch_size=args.batch, shuffle=True)

            for batch in iterator_train:
                ques         = batch['ques'].to(args.device)
                ques_len     = batch['ques_len']
                contx        = batch['contx'].to(args.device)
                contx_len    = batch['contx_len']
                ans          = batch['ans1'].to(args.device)
                ans_len      = batch['ans1_len']
                ans_mask     = batch['ans1_mask'].to(args.device)
                ans1_tok_idx = batch['ans1_tok_idx'].to(args.device)

                optimizer.zero_grad()

                pred        = model(ques, ques_len, contx, contx_len,
                                    ans, ans_len, ans_mask, is_inferring=False)
                # pred: [batch, seq_len_ans, d_vocab]
                pred_flat   = pred.view(-1, args.d_vocab)
                ans1_flat   = ans1_tok_idx.view(-1)


                loss        = criterion(pred_flat, ans1_flat)

                loss.backward()
                torch_nn.utils.clip_grad_value_(model.parameters(), clip_value=0.25)
                optimizer.step()
                scheduler.step()

                loss_train += loss.detach().item()

                logging.info(f"  train: batch {nth_batch:4d} | loss: {loss:8.6f}")
                nth_batch += 1

        return loss_train / nth_batch


    def test(self, model, dataset_test, criterion):
        nth_batch, n_samples            = 0, 0
        loss_test                       = 0
        bleu_1, bleu_4, meteor, rouge_l = 0, 0, 0, 0

        model.eval()

        with torch.no_grad():
            for eval_file in dataset_test.file_names:
                logging.info(f"Eval with file: {eval_file}")

                dataset_test.read_shard(eval_file)
                iterator_test  = DataLoader(dataset_test, batch_size=args.batch)
                n_samples   += len(dataset_test)

                for batch in iterator_test:
                    ques        = batch['ques'].to(args.device)
                    ques_len    = batch['ques_len']
                    contx       = batch['contx'].to(args.device)
                    contx_len   = batch['contx_len']
                    ans         = batch['ans1'].to(args.device)
                    ans_len     = batch['ans1_len']
                    ans_mask    = batch['ans1_mask'].to(args.device)
                    ans1_tok_idx = batch['ans1_tok_idx'].to(args.device)
                    ans2_tok_idx = batch['ans2_tok_idx'].to(args.device)

                    pred        = model(ques, ques_len, contx, contx_len,
                                        ans, ans_len, ans_mask, is_inferring=True)
                    # pred: [batch, seq_len_ans, d_vocab]
                    pred_flat   = pred.view(-1, args.d_vocab)
                    ans1_flat   = ans1_tok_idx.view(-1)

                    loss        = criterion(pred_flat, ans1_flat)

                    loss_test += loss.detach().item()

                    bleu_1_, bleu_4_, meteor_, rouge_l_ = self.get_batch_scores(pred, ans1_tok_idx, ans2_tok_idx)

                    bleu_1  += bleu_1_
                    bleu_4  += bleu_4_
                    meteor  += meteor_
                    rouge_l += rouge_l_

                    logging.info(f"  test: batch {nth_batch:4d} | loss: {loss:8.6f}")
                    nth_batch += 1

        with open(PATH["eval_result"], 'a+') as result_file:
            json.dump({
                'tag'   : "infer",
                'bleu_1': bleu_1/n_samples,
                'bleu_4': bleu_4/n_samples,
                'meteor': meteor/n_samples,
                'rouge_l': rouge_l/n_samples,
            }, result_file, indent=2, ensure_ascii=False)

        return loss_test / nth_batch


    def trigger_train(self):
        ###############################
        # Load data
        ###############################
        dataset_train   = CustomDataset(os.path.dirname(PATH['dataset_para']).replace("[SPLIT]", "train"),
                                       PATH['vocab'])
        dataset_test    = CustomDataset(os.path.dirname(PATH['dataset_para']).replace("[SPLIT]", "test"),
                                      PATH['vocab'])


        ###############################
        # Defind model and associated stuffs
        ###############################
        model       = NarrativePipeline(self.vocab).to(args.device)
        optimizer   = AdamW(model.parameters(), lr=args.lr, weight_decay=args.w_decay)
        scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)

        criterion   = torch_nn.CrossEntropyLoss(ignore_index=self.vocab.stoi(self.vocab.PAD))

        for p in model.parameters():
            if p.dim() > 1:
                torch_nn.init.xavier_uniform_(p)

        start_epoch     = 0
        best_loss_test  = 10e10

        # Check if previous checkpoint available
        if check_exist(PATH['saved_model']):
            logging.info("=> Saved model instance existed. Load it.")

            states  = self.load_checkpoint()

            model.load_state_dict(states['model_state'])
            optimizer.load_state_dict(states['optim_state'])
            scheduler.load_state_dict(states['sched_state'])
            start_epoch     = states['epoch']
            best_loss_test  = states['best_loss_test']


        ###############################
        # Start training
        ###############################

        for nth_epoch in range(start_epoch, args.n_epochs):
            logging.info(f"= Epoch {nth_epoch}")

            loss_train  = self.train(model, dataset_train, criterion, optimizer, scheduler)
            loss_test   = self.test(model, dataset_test, criterion)

            logging.info("Switch Answerability.")
            dataset_train.switch_answerability()
            if loss_test < best_loss_test:
                best_loss_test = loss_test
                self.save_model(model)
            self.save_model(model)

            logging.info(f"= Epoch {nth_epoch} finishes: loss_train {loss_train:.5f} | loss_test {loss_test:.5f}")
            self.save_checkpoint(model, optimizer, scheduler, nth_epoch, best_loss_test)


    def get_batch_scores(self, pred, ans1_tok_idx, ans2_tok_idx) -> tuple:
        """Calculate BLEU-1, BLEU4, METEOR and ROUGE_L for each entry in batch.

        Args:
            pred (tensor): predicted tensor
            ans_tok_idx (tensor): target answer
        """
        # pred: [batch, d_vocab + seq_len_cntx, seq_len_ans]
        # ans_tok_idx: [batch, seq_len_ans]

        pred_   = torch.argmax(torch_f.log_softmax(pred, 1), 1)
        # pred_: [batch, seq_len_ans]

        bleu_1, bleu_4, meteor, rouge_l = 0, 0, 0, 0
        # Calculate for each batch
        for p, a1, a2 in zip(pred_.tolist(), ans1_tok_idx.tolist(), ans2_tok_idx.tolist()):
            p  = ' '.join([self.vocab.itos(i) for i in p  if i > 0])
            a1 = ' '.join([self.vocab.itos(i) for i in a1 if i > 0])
            a2 = ' '.join([self.vocab.itos(i) for i in a2 if i > 0])

            bleu_1_, bleu_4_, meteor_, rouge_l_ = get_scores([a1, a2], p)

            bleu_1  += bleu_1_
            bleu_4  += bleu_4_
            meteor  += meteor_
            rouge_l += rouge_l_

        return bleu_1, bleu_4, meteor, rouge_l


    def trigger_infer(self):
        ###############################
        # Load data
        ###############################
        dataset_valid   = CustomDataset(os.path.dirname(PATH['dataset_para']).replace("[SPLIT]", "validation"),
                                      PATH['vocab'])

        ###############################
        # Defind model and associated stuffs
        ###############################
        model       = NarrativePipeline(self.vocab).to(args.device)
        criterion   = torch_nn.CrossEntropyLoss()
        model       = model.to(args.device)

        model       = self.load_model(model)

        ###############################
        # Start infering
        ###############################
        nth_batch, n_samples            = 0, 0
        loss_test                       = 0
        bleu_1, bleu_4, meteor, rouge_l = 0, 0, 0, 0

        model.eval()

        with torch.no_grad():
            for valid_file in dataset_valid.file_names:
                logging.info(f"Valid with file: {valid_file}")

                dataset_valid.read_shard(valid_file)
                n_samples   += len(dataset_valid)

                iterator_valid  = DataLoader(dataset_valid, batch_size=args.batch)
                for batch in iterator_valid:
                    ques        = batch['ques'].to(args.device)
                    ques_len    = batch['ques_len']
                    contx       = batch['contx'].to(args.device)
                    contx_len   = batch['contx_len']
                    ans         = batch['ans1'].to(args.device)
                    ans_len     = batch['ans1_len']
                    ans_mask    = batch['ans1_mask'].to(args.device)
                    ans1_tok_idx = batch['ans1_tok_idx'].to(args.device)
                    ans2_tok_idx = batch['ans2_tok_idx'].to(args.device)

                    pred        = model(ques, ques_len, contx, contx_len,
                                        ans, ans_len, ans_mask, is_inferring=True)
                    # pred: [batch, seq_len_ans, d_vocab]
                    pred_flat   = pred.view(-1, args.d_vocab)
                    ans1_flat   = ans1_tok_idx.view(-1)

                    loss        = criterion(pred_flat, ans1_flat)

                    # Calculate loss
                    loss_test   += loss.detach().item()

                    # Get batch score
                    bleu_1_, bleu_4_, meteor_, rouge_l_ = self.get_batch_scores(pred, ans1_tok_idx, ans2_tok_idx)

                    bleu_1  += bleu_1_
                    bleu_4  += bleu_4_
                    meteor  += meteor_
                    rouge_l += rouge_l_

                    logging.info(f"  validate: batch {nth_batch} | loss: {loss:8.6f}")
                    nth_batch += 1


        logging.info(f"End validattion: bleu_1 {bleu_1/n_samples:.5f} | bleu_4 {bleu_4/n_samples:.5f} \
                     | meteor {meteor/n_samples:.5f} | rouge_l {rouge_l/n_samples:.5f}")

        with open(PATH["eval_result"], 'a+') as result_file:
            json.dump({
                'tag'   : "infer",
                'bleu_1': bleu_1/n_samples,
                'bleu_4': bleu_4/n_samples,
                'meteor': meteor/n_samples,
                'rouge_l': rouge_l/n_samples,
            }, result_file, indent=2, ensure_ascii=False)

if __name__ == '__main__':
    logging.info("* Start NarrativePipeline")


    narrative_pipeline  = Trainer()

    narrative_pipeline.trigger_train()

    narrative_pipeline.trigger_infer()
