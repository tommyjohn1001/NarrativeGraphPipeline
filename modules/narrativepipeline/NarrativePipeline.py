import os, json
from typing import List

from torch.utils.data import DataLoader
import torch.nn as torch_nn
import torch
from transformers import AdamW


from modules.narrativepipeline.utils import CustomDataset, build_vocab, Vocab
from modules.ans_infer.Transformer import TransDecoder
from modules.utils import check_exist
from modules.reasoning.MemoryBased import MemoryBasedReasoning
from modules.finegrained.FineGrain import FineGrain
from configs import args, logging, PATH


class  NarrativePipeline(torch_nn.Module):
    def __init__(self, vocab):
        super().__init__()

        self.embd_layer = FineGrain()
        self.reasoning  = MemoryBasedReasoning()
        self.ans_infer  = TransDecoder(vocab, self.embd_layer.embedding)

    def forward(self, ques, ques_mask, ans, ans_mask,
                paras, paras_mask, is_inferring=False):
        # ques       : [b, seq_len_ques]
        # ques_mask  : [b, seq_len_ques]
        # paras      : [b, n_paras, seq_len_para]
        # paras_mask : [b, n_paras, seq_len_para]
        # ans        : [b, seq_len_ans]
        # ans_mask   : [b, seq_len_ans]
        # NOTE: all arguments used in this method must not modify the orginal

        ####################
        # Embed question and context with FineGrain
        ####################
        ques, paras, ans = self.embd_layer(ques, paras, ans,
                                           ques_mask, paras_mask, ans_mask)
        # ques  : [b, seq_len_ques, d_hid]
        # paras : [b, n_paras, seq_len_para, d_hid]
        # ans   : [b, seq_len_ans, d_hid]


        ####################
        # Do reasoning with IAL
        ####################
        Y       = self.reasoning(ques, paras)
        # Y: [b, seq_len_ques_para=seq_len_ques+seq_len_para, d_hid]


        ####################
        # Generate answer with PGD
        ####################
        pred    = self.ans_infer(Y, ans, ans_mask, is_inferring)
        # pred: [batch, seq_len_ans, d_vocab]

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

        self.first_time = True

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

    def save_checkpoint(self, model: NarrativePipeline, optimizer, scheduler, epoch, best_loss_test):
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

        ## Save tensors belonging to MemoryModule
        model.reasoning.save_memory()

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
                ques_mask    = batch['ques_mask'].to(args.device)
                ans1         = batch['ans1'].to(args.device)
                ans1_mask    = batch['ans1_mask'].to(args.device)
                ans1_loss    = batch['ans1_loss'].to(args.device)
                paras        = batch['paras'].to(args.device)
                paras_mask   = batch['paras_mask'].to(args.device)

                optimizer.zero_grad()

                pred        = model(ques, ques_mask, ans1, ans1_mask,
                                    paras, paras_mask, is_inferring=False)
                # pred: [batch, seq_len_ans, d_vocab]
                pred_flat   = pred.view(-1, args.d_vocab)
                ans1_flat   = ans1_loss.view(-1)

                loss        = criterion(pred_flat, ans1_flat)
                loss.backward()

                torch_nn.utils.clip_grad_value_(model.parameters(), clip_value=1)
                optimizer.step()
                scheduler.step()

                loss_train += loss.detach().item()

                logging.info(f"  train: batch {nth_batch:4d} | loss: {loss:8.6f}")
                nth_batch += 1

        return loss_train / nth_batch


    def test(self, model, dataset_test, criterion):
        nth_batch, n_samples            = 0, 0
        loss_test                       = 0

        model.eval()

        with torch.no_grad():
            for eval_file in dataset_test.file_names:
                logging.info(f"Test with file: {eval_file}")

                dataset_test.read_shard(eval_file)
                iterator_test  = DataLoader(dataset_test, batch_size=args.batch)
                n_samples   += len(dataset_test)

                for batch in iterator_test:
                    ques         = batch['ques'].to(args.device)
                    ques_mask    = batch['ques_mask'].to(args.device)
                    ans1         = batch['ans1'].to(args.device)
                    ans2         = batch['ans2'].to(args.device)
                    ans1_mask    = batch['ans1_mask'].to(args.device)
                    ans1_loss    = batch['ans1_loss'].to(args.device)
                    paras        = batch['paras'].to(args.device)
                    paras_mask   = batch['paras_mask'].to(args.device)

                    pred        = model(ques, ques_mask, ans1, ans1_mask,
                                        paras, paras_mask, is_inferring=True)
                    # pred: [batch, seq_len_ans, d_vocab]
                    pred_flat   = pred.view(-1, args.d_vocab)
                    ans1_flat   = ans1_loss.view(-1)

                    loss        = criterion(pred_flat, ans1_flat)

                    loss_test += loss.detach().item()


                    logging.info(f"  test: batch {nth_batch:4d} | loss: {loss:8.6f}")
                    nth_batch += 1

        return loss_test / nth_batch


    def trigger_train(self):
        ###############################
        # Load data
        ###############################
        dataset_train   = CustomDataset(os.path.dirname(PATH['dataset_para']).replace("[SPLIT]", "train"),
                                       self.vocab)
        dataset_test    = CustomDataset(os.path.dirname(PATH['dataset_para']).replace("[SPLIT]", "test"),
                                       self.vocab)


        ###############################
        # Defind model and associated stuffs
        ###############################
        model       = NarrativePipeline(self.vocab).to(args.device)
        optimizer   = AdamW(model.parameters(), lr=args.lr, weight_decay=args.w_decay)
        scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)

        criterion   = torch_nn.CrossEntropyLoss(ignore_index=self.vocab.stoi(self.vocab.pad))

        for p in model.parameters():
            if p.dim() > 1:
                torch_nn.init.xavier_uniform_(p)

        start_epoch = 0
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

            logging.info(f"= Epoch {nth_epoch} finishes: loss_train {loss_train:.5f} | loss_test {loss_test:.5f}")
            self.save_checkpoint(model, optimizer, scheduler, nth_epoch, best_loss_test)

    def clean_sent(self, sent: List[int]):
        sent_ = []
        for tok in sent:
            if tok not in [self.vocab.cls_id, self.vocab.sep_id, self.vocab.unk_id, self.vocab.sep_id]:
                sent_.append(self.vocab.itos(tok))

        return ' '.join(sent_)

    def trigger_infer(self):
        ###############################
        # Load data
        ###############################
        dataset_valid   = CustomDataset(os.path.dirname(PATH['dataset_para']).replace("[SPLIT]", "validation"),
                                      self.vocab)

        ###############################
        # Defind model and associated stuffs
        ###############################
        model       = NarrativePipeline(self.vocab).to(args.device).to(args.device)
        model       = self.load_model(model)

        ###############################
        # Start infering
        ###############################
        pred_result = []

        model.eval()

        with torch.no_grad():
            for valid_file in dataset_valid.file_names:
                logging.info(f"Valid with file: {valid_file}")

                dataset_valid.read_shard(valid_file)

                iterator_valid  = DataLoader(dataset_valid, batch_size=args.batch)
                for batch in iterator_valid:
                    ques         = batch['ques'].to(args.device)
                    ques_mask    = batch['ques_mask'].to(args.device)
                    ans1         = batch['ans1'].to(args.device)
                    ans2         = batch['ans2'].to(args.device)
                    ans1_mask    = batch['ans1_mask'].to(args.device)
                    paras        = batch['paras'].to(args.device)
                    paras_mask   = batch['paras_mask'].to(args.device)

                    pred        = model(ques, ques_mask, ans1, ans1_mask,
                                        paras, paras_mask, is_inferring=True)
                    for pred_, ans1_, ans2_ in zip(pred, ans1, ans2):
                        pred_result.append({
                            'pred': self.clean_sent(pred_),
                            'ans1': self.clean_sent(ans1_),
                            'ans2': self.clean_sent(ans2_),
                        })

                    with open(PATH['prediction'], 'a+') as result_file:
                        json.dump(pred_result, result_file, indent=2, ensure_ascii=False)

if __name__ == '__main__':
    logging.info("* Start NarrativePipeline")


    narrative_pipeline  = Trainer()

    narrative_pipeline.trigger_train()

    narrative_pipeline.trigger_infer()
