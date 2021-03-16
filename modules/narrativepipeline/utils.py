'''This file contain modules for loading data and training procedure. Other component layers
are in other directories.'''

import glob, ast

import torch.nn. functional as torch_f
from torch.utils.data import Dataset
from transformers import BertTokenizer
from bs4 import BeautifulSoup
import pandas as pd
import spacy
import torch

from configs import args

tokenizer   = BertTokenizer.from_pretrained(args.bert_model)
nlp         = spacy.load("en_core_web_sm", disable=['ner', 'parser', 'tagger', "lemmatizer"])


# NOTE: This class is to create Dataset instance for training. This class belongs to my proposal pipeline.
class CustomDataset(Dataset):
    def __init__(self, path_csv_dir):
        file_names = glob.glob(f"{path_csv_dir}/data_*.csv")
        
        dfs = []
        for filename in file_names:
            df = pd.read_csv(filename, index_col=None, header=0)
            dfs.append(df)

        self.df = pd.concat(dfs, axis=0, ignore_index=True)


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if isinstance(idx, int):
            return self.cvt_tensor(self.df.iloc[idx])

        return [self.cvt_tensor(entry) for entry in self.df.iloc[idx].itertuples()]

    def cvt_tensor(self, entry):
        """ Convert entry to tensors """

        ## Process question: tokenize, pad and question mask
        ques        = entry.question.split(' ')
        ques        = torch.LongTensor(tokenizer.convert_tokens_to_ids(ques))
        ques_mask   = torch.ones(ques.shape[0])

        ques_mask   = torch_f.pad(ques_mask, (0, args.seq_len_ques - ques.shape[0]))
        ques        = torch_f.pad(ques, (0, args.seq_len_ques - ques.shape[0]))


        ## Process each paragraph in En: tokenize, pad, create mask
        list_para_batch, list_para_mask_batch = [], []
        paras   = ast.literal_eval(entry.En)
        for para in paras:
            para        = BeautifulSoup(para, 'html.parser').get_text()
            para        = [tok.text for tok in nlp(para)]

            para_mask   = torch.ones(len(para))
            para        = torch.LongTensor(tokenizer.convert_tokens_to_ids(para))

            para_mask   = torch_f.pad(para_mask, (0, args.seq_len_para - para.shape[0]))
            para        = torch_f.pad(para, (0, args.seq_len_para - para.shape[0]))


            list_para_batch.append(para)
            list_para_mask_batch.append(para_mask)

        para_En         = torch.vstack(list_para_batch).long()
        para_En_mask    = torch.vstack(list_para_mask_batch).long()

        para_En_mask    = torch_f.pad(para_En, (0, 0, 0, args.n_paras - para_En.shape[0]))
        para_En         = torch_f.pad(para_En, (0, 0, 0, args.n_paras - para_En.shape[0]))


        ## Process each paragraph in Hn: tokenize, pad, create mask
        list_para_batch, list_para_mask_batch = [], []
        paras   = ast.literal_eval(entry.En)
        for para in paras:
            para        = BeautifulSoup(para, 'html.parser').get_text()
            para        = [tok.text for tok in nlp(para)]

            para_mask   = torch.ones(len(para))
            para        = torch.LongTensor(tokenizer.convert_tokens_to_ids(para))

            para_mask   = torch_f.pad(para_mask, (0, args.seq_len_para - para.shape[0]))
            para        = torch_f.pad(para, (0, args.seq_len_para - para.shape[0]))


            list_para_batch.append(para)
            list_para_mask_batch.append(para_mask)

        para_Hn         = torch.vstack(list_para_batch).long()
        para_Hn_mask    = torch.vstack(list_para_mask_batch).long()

        para_Hn_mask    = torch_f.pad(para_Hn, (0, 0, 0, args.n_paras - para_Hn.shape[0]))
        para_Hn         = torch_f.pad(para_Hn, (0, 0, 0, args.n_paras - para_Hn.shape[0]))


        ## Process answers
        answer1 = tokenizer.convert_tokens_to_ids(entry.answers[0].split(' '))
        answer2 = tokenizer.convert_tokens_to_ids(entry.answers[1].split(' '))

        answer1 = torch_f.pad(torch.LongTensor(answer1), (0, args.seq_len_ans - len(answer1)))
        answer2 = torch_f.pad(torch.LongTensor(answer2), (0, args.seq_len_ans - len(answer2)))


        return {
            'question'      : ques,
            'question_mask' : ques_mask,
            'para_En'       : para_En,
            'para_En_mask'  : para_En_mask,
            'para_Hn'       : para_Hn,
            'para_Hn_mask'  : para_Hn_mask,
            'answer1'       : answer1,
            'answer2'       : answer2
        }
