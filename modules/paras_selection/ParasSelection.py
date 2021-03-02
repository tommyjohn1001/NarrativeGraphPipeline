from glob import glob
import json, os

from transformers import BertModel, AdamW, get_linear_schedule_with_warmup,\
    BertTokenizer
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as torch_f
import torch.nn as torch_nn
import torch
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
import spacy

from modules.paras_selection.utils import pad
from modules.utils import check_file_existence
from configs import args, logging, PATH

FILTERING_THETA = 0.1

Tokenizer   = BertTokenizer.from_pretrained(args.bert_model)
nlp         = spacy.load("en_core_web_sm")

CLS = Tokenizer.cls_token_id
SEP = Tokenizer.sep_token_id
PAD = Tokenizer.pad_token_id

MAX_QUESTION_LEN    = 34
MAX_PARA_LEN        = 475

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

class CustomDataset(Dataset):
    def __init__(self, path) -> None:
        super().__init__()

        assert os.path.isfile(path), f"File {path} not existed."

        self.data_csv   = pd.read_csv(path).drop(columns=['Unnamed: 0']).applymap(lambda x: json.loads(str(x)))

    def __len__(self):
        return len(self.data_csv)

    def __getitem__(self, index):
        return self.data_csv.iloc[index].to_dict()

class ParasSelection:
    ## TODO: Test this class

    #########################################################
    # METHODS FOR TRAINING PROCESS
    #########################################################
    def train(self, dataloader, model, optimizer, loss, scheduler):
        model.train()

        accum_loss  = 0
        for batch in tqdm(dataloader):
            srcs    = torch.vstack(batch['src']).transpose(0, 1).to(args.device)
            masks   = torch.vstack(batch['mask']).transpose(0, 1).to(args.device)
            trgs    = torch.vstack(batch['trg']).transpose(0, 1).to(args.device)

            predict     = model(srcs, masks)

            output      = loss(predict, trgs)

            output.backward()
            optimizer.step()
            scheduler.step()

            accum_loss  += output


        return accum_loss / len(dataloader)


    def eval(self, dataloader, model, loss):
        model.eval()

        with torch.no_grad():
            accum_loss  = 0
            for  batch in tqdm(dataloader):
                srcs    = torch.vstack(batch['src']).transpose(0, 1).to(args.device)
                masks   = torch.vstack(batch['mask']).transpose(0, 1).to(args.device)
                trgs    = torch.vstack(batch['trg']).transpose(0, 1).to(args.device)

                predict     = model(srcs, masks)

                output      = loss(predict, trgs)

                accum_loss  += output

        return accum_loss / len(dataloader)


    def trigger_train(self):
        """Trigger trainning Paras Selection model
        """

        #######################
        ### Read data for training and test
        #######################
        logging.info("1. Load train, test and eval data")

        path            = glob("./backup/data_parasselection/train.csv")
        dataset_train   = CustomDataset(path)

        path            = glob("./backup/data_parasselection/test.csv")
        dataset_test    = CustomDataset(path)

        path            = glob("./backup/data_parasselection/valid.csv")
        dataset_valid   = CustomDataset(path)


        #######################
        ### Specify model
        #######################
        logging.info("2. Declare model, optimizer, etc")

        model       = ParasScoring().to(args.device)
        optimizer   = AdamW(model.parameters())
        scheduler   = get_linear_schedule_with_warmup(optimizer, 1000, len(dataset_train))
        loss        = torch_nn.MSELoss()


        #######################
        ### Start training
        #######################
        logging.info("3. Start training and testing")

        dataloader_train    = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
        dataloader_test     = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=True)
        dataloader_valid    = DataLoader(dataset_valid, batch_size=args.batch_size, shuffle=True)

        best_lost_test  = 10e5
        for n_epoch in range(args.n_epochs):
            loss_train  = self.train(dataloader_train, model, optimizer, loss, scheduler)
            loss_test   = self.eval(dataloader_test, model, loss)

            if loss_test < best_lost_test:
                best_lost_test = loss_test
                self.save_model(model)

            logging.info(f"epoch {n_epoch:3d} | train: loss: {loss_train:10.5f} | eval: loss: {loss_test:10.5f}")


        #######################
        ### Evaluate
        #######################
        logging.info("4. Start evaluating model")

        loss_eval   = self.eval(dataloader_valid, model, loss)
        logging.info(f"=> Eval: loss: {loss_eval:7.3f}")


    #########################################################
    # METHODS FOR INFERENCE PROCESS
    #########################################################
    def f_infer(self, document: dict, model) -> dict:
        """Function to be fed into mapping function of each dataset.
        This is used only for inferring

        Args:
            document (dict): each entry of dataset
            model : model used for inferring

        Returns:
            dict: entry after being updated
        """
        ## Initial preparation
        question        = json.loads(document['question_tokens'])
        paragraphs      = json.loads(document['doc_tokens'])

        question_       = pad(question, MAX_QUESTION_LEN)

        list_paragraphs = []    ## This list holds paragraphs whose score
                                ## is greater than FILTERING_THETA

        ## For each paragraph, convert into tensors,
        ## pass tensors to model, calculate mean score of
        ## output from those tensor and compare mean score with
        for paragraph in paragraphs:
            list_pairs  = []

            ### Create pairs
            for ith in range(0, len(paragraph), MAX_PARA_LEN):
                paragraph_  = pad(paragraph[ith:ith+MAX_PARA_LEN], MAX_PARA_LEN)

                pair    = [CLS] + question_[0] + [SEP] + paragraph_[0] + [SEP]
                mask    = [1]   + question_[1] + [1]   + paragraph_[1] + [1]

                list_pairs.append({
                    'src'   : pair,
                    'mask'  : mask,
                })

            ### Batchify pairs
            src, mask = [], []
            for pair in list_pairs:
                src.append(pair['src'])
                mask.append(pair['mask'])

            ## Start inferring with model
            model.eval()
            with torch.no_grad():
                for ith in range(0, len(src), args.batch_size):
                    src_    = torch.vstack(src[ith:ith+args.batch_size]).to(args.device)
                    mask_   = torch.vstack(mask[ith:ith+args.batch_size]).to(args.device)

                    predict = model(src_, mask_)

                    if torch.mean(predict) >= FILTERING_THETA:
                        list_paragraphs.append(paragraph)


        document['doc_tokens']  = list_paragraphs

        return document


    def trigger_inference(self):
        """Trigger inferring Paras Selection model
        """

        ## Read data for train, test and valid
        dataset_train   = load_dataset('csv', data_files=glob('backup/processed_data/train/*.csv'))
        dataset_test    = load_dataset('csv', data_files=glob('backup/processed_data/test/*.csv'))
        dataset_valid   = load_dataset('csv', data_files=glob('backup/processed_data/valid/*.csv'))


        ## Specify model together with load model
        model   = ParasScoring().to(args.device)

        assert check_file_existence(PATH['savemodel_ParasSelection']), f"Model not found in {PATH['savemodel_ParasSelection']}"
        model.load_state_dict(torch.load(PATH['savemodel_ParasSelection']))


        ## Infer datasets
        dataset_train   = dataset_train.map(self.f_infer, remove_columns=['goldeness'])
        dataset_test    = dataset_test.map(self.f_infer, remove_columns=['goldeness'])
        dataset_valid   = dataset_valid.map(self.f_infer, remove_columns=['goldeness'])


        ## Infer dataset
        dataset_train.to_csv(PATH['data'].replace("[SPLIT]", "train"))
        dataset_test.to_csv(PATH['data'].replace("[SPLIT]", "test"))
        dataset_valid.to_csv(PATH['data'].replace("[SPLIT]", "valid"))


    #########################################################
    # MISCELLANEOUS METHODS
    #########################################################
    def save_model(self, model: torch_nn.Module):
        """Save trained model

        Args:
            model (torch_nn.Module): model to be saved
        """

        ### Try to create folder if not existed
        try:
            os.makedirs(os.path.dirname(PATH['savemodel_ParasSelection']))
        except FileExistsError:
            pass

        torch.save(model.state_dict(), PATH['savemodel_ParasSelection'])
     