from glob import glob
import os


from transformers import BertModel, AdamW, get_linear_schedule_with_warmup
from datasets import load_dataset, Dataset
# from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as torch_f
import torch.nn as torch_nn
import torch

from modules.paras_selection.utils import create_tensors
from configs import args, logging, PATH



class ParasScoring(torch_nn.Module):
    """ Score paras
    """

    def __init__(self, hidden_size:int = 768, max_seq_len:int = 512):
        super().__init__()

        self.embedding  = BertModel.from_pretrained(PATH['bert_model'])
        self.linear     = torch_nn.Linear(hidden_size, 1)
        self.linear2    = torch_nn.Linear(max_seq_len, 1)

    def forward(self, src: torch.Tensor, attn_mask: torch.Tensor) -> torch.tensor:
        ## src, attn_mask: [batch, seq_len]

        X   = self.embedding(src, attn_mask)[0]
        ## X: [batch, seq_len, hidden_size]

        X   = self.linear(X)

        X   = torch_f.dropout(X, 0.2)
        ## X: [batch, seq_len, 1]

        X   = torch.squeeze(X)
        X   = self.linear2(X)
        ## X: [batch, 1]

        return torch.squeeze(X)



class ParasSelection:
    ## TODO: Test this class

    #########################################################
    # METHODS FOR TRAINING PROCESS
    #########################################################
    def train(self, dataset: Dataset, model, optimizer, loss, scheduler):
        model.train()

        accum_loss  = 0
        n_samples  = 0
        for ith_batch in range(args.batch_size):
            ### Shard data and processs data to tensors
            batched_dataset = dataset.shard(args.batch_size, ith_batch)

            ### Create data tensors from processed dataset
            src         = torch.tensor(batched_dataset['src']).to(args.device)
            attn_mask   = torch.tensor(batched_dataset['attn_mask']).to(args.device)
            trg         = torch.tensor(batched_dataset['trg']).to(args.device)

            ### Push tensors to model to train
            predict     = model(src, attn_mask)

            ### Calculate loss and do some stuffs
            output      = loss(predict, trg)

            output.backward()
            optimizer.step()
            scheduler.step()

            n_samples  += batched_dataset.num_rows
            accum_loss  += output

        return accum_loss / n_samples


    def eval(self, dataset, model, loss):
        model.eval()

        with torch.no_grad():
            accum_loss, n_samples   = 0, 0
            for ith_batch in range(args.batch_size):
                ### Shard data and processs data to tensors
                batched_dataset = dataset.shard(args.batch_size, ith_batch)

                ## TODO: Test this method
                batched_dataset = batched_dataset.map(create_tensors, num_proc=args.num_proc,
                                                    remove_columns=[batched_dataset.column_names()])

                ### Create data tensors from processed dataset
                src         = torch.tensor(batched_dataset['src']).to(args.device)
                attn_mask   = torch.tensor(batched_dataset['attn_mask']).to(args.device)
                trg         = torch.tensor(batched_dataset['trg']).to(args.device)

                ### Push tensors to model to train
                predict     = model(src, attn_mask)

                ### Calculate loss and do some stuffs
                output      = loss(predict, trg)

                n_samples   += batched_dataset.num_rows
                accum_loss  += output

        return accum_loss / n_samples


    def trigger_train(self):
        """Trigger trainning Paras Selection model
        """

        #######################
        ### Read data for training and test
        #######################
        logging.info("1. Load train, test and eval data")

        paths           = glob("./backup/processed_data/train")
        dataset_train   = load_dataset('pandas', data_files=paths)['train']

        paths           = glob("./backup/processed_data/test")
        dataset_test    = load_dataset('pandas', data_files=paths)['train']

        paths           = glob("./backup/processed_data/eval")
        dataset_eval    = load_dataset('pandas', data_files=paths)['train']


        #######################
        ### Specify model
        #######################
        logging.info("2. Declare model, optimizer, etc")

        model       = ParasScoring().to(args.device)
        optimizer   = AdamW(model.parameters())
        scheduler   = get_linear_schedule_with_warmup(optimizer, 1000, dataset_train)
        loss        = torch_nn.MSELoss()


        #######################
        ### Start training
        #######################
        logging.info("3. Start training and testing")

        best_lost_test  = 10e5
        for n_epoch in range(args.n_epochs):
            loss_train  = self.train(dataset_train, model, optimizer, loss, scheduler)

            loss_test   = self.eval(dataset_test, model, loss)

            if loss_test < best_lost_test:
                best_lost_test = loss_test
                self.save_model(model)

            logging.info(f"epoch {n_epoch:2d} | train: loss: {loss_train:7.3f} | eval: loss: {loss_test:7.3f}")


        #######################
        ### Evaluate
        #######################
        logging.info("4. Start evaluating model")

        loss_eval   = self.eval(dataset_eval, model, loss)
        logging.info(f"=> Eval: loss: {loss_eval:7.3f}")


    #########################################################
    # METHODS FOR INFERENCE PROCESS
    #########################################################
    def trigger_inference(self):
        """Trigger inferring Paras Selection model
        """

        ## TODO: Continue implementing this method

        #######################
        ### Read data for training and test
        #######################
        ## TODO: Read data from folder 'processed_data'


        #######################
        ### Specify model
        #######################
        ## TODO: Load model from saved
        model       = None
        loss        = torch_nn.MSELoss()


        #######################
        ### Start infering
        #######################
        ## Infer for every entry of dataset
        ## TODO: Loop every entry of train, test and valid dataset, decompose into tensors using function 'create_tensors' and feed into mode


    #########################################################
    # MISCELLANEOUS METHODS
    #########################################################
    def save_model(self, model: torch_nn.Module):
        """Save trained model

        Args:
            model (torch_nn.Module): model to be saved
        """

        #######################
        ### Try to create folder
        ### if not existed
        #######################
        try:
            os.makedirs(os.path.dirname(PATH['savemodel_ParasSelection']))
        except FileExistsError:
            pass

        #######################
        ### Try to create folder
        ### if not existed
        #######################
        torch.save(model.state_dict(), PATH['savemodel_ParasSelection'])
