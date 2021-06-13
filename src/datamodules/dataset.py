from random import sample
import glob, ast, gc

from torch.utils.data import Dataset
import torch
from transformers import BertTokenizer
import pandas as pd
import numpy as np

from src.utils.utils import ParallelHelper


class NarrativeDataset(Dataset):
    def __init__(
        self,
        split: str,
        path_data: str,
        path_bert: str,
        size_dataset: int,
        size_shard: int,
        seq_len_ques: int = 42,
        seq_len_para: int = 162,
        seq_len_ans: int = 42,
        n_paras: int = 30,
        num_worker: int = 1,
    ):

        self.split = split
        self.seq_len_ques = seq_len_ques
        self.seq_len_para = seq_len_para
        self.seq_len_ans = seq_len_ans
        self.n_paras = n_paras
        self.num_workers = num_worker
        self.size_dataset = size_dataset
        self.size_shard = size_shard

        path_data = path_data.replace("[SPLIT]", split).replace("[SHARD]", "*")
        self.paths = sorted(glob.glob(path_data))

        ## NOTE: Instead of own vocab, BERT vocab is used
        # self.tokenizer = CustomBertTokenizer(path_vocab)
        self.tokenizer = BertTokenizer.from_pretrained(path_bert)

        self.curent_ith_file = -1

        self.ques = None
        self.ques_mask = None
        self.ans1 = None
        self.ans2 = None
        self.ans1_mask = None
        self.paras = None
        self.paras_mask = None

        self.exchange_rate = 0

    def __len__(self) -> int:
        return self.size_dataset

    def __getitem__(self, indx):
        if torch.is_tensor(indx):
            indx = indx.tolist()

        ith_file = indx // self.size_shard
        indx = indx % self.size_shard

        # Check nth file and reload dataset if needed
        if ith_file != self.curent_ith_file:
            self.curent_ith_file = ith_file

            # Reload dataset
            gc.collect()
            self.read_datasetfile(self.paths[self.curent_ith_file])

        return {
            # 'docId'         : self.docId[indx],
            # 'ques_plain'    : self.ques_plain[indx],
            "ques": self.ques[indx],
            "ques_mask": self.ques_mask[indx],
            "ans1": self.ans1[indx],
            "ans2": self.ans2[indx],
            "ans1_mask": self.ans1_mask[indx],
            "paras": self.paras[indx],
            "paras_mask": self.paras_mask[indx],
        }

    def f_process_file_multi(self, entries, queue):
        for entry in entries.itertuples():
            queue.put(self.f_process_file_single(entry))

    def f_process_file_single(self, entry):
        ###########################
        # Process question
        ###########################
        encoded = self.tokenizer(
            entry.question,
            padding="max_length",
            max_length=self.seq_len_ques,
            truncation=True,
            return_tensors="np",
            return_token_type_ids=False,
        )
        ques = encoded["input_ids"][0]
        ques_mask = encoded["attention_mask"][0]

        ###########################
        # Process answers
        ###########################
        answers = ast.literal_eval(entry.answers)

        # This trick ensures training process occurs in longer answer
        if len(" ".split(answers[0])) < len(" ".split(answers[1])):
            answers[0], answers[1] = answers[1], answers[0]

        ans1, ans2 = answers

        encoded = self.tokenizer(
            ans1,
            padding="max_length",
            truncation=True,
            max_length=self.seq_len_ans,
            return_tensors="np",
            return_token_type_ids=False,
        )
        ans1 = encoded["input_ids"][0]
        ans1_mask = encoded["attention_mask"][0]

        encoded = self.tokenizer(
            ans2,
            padding="max_length",
            truncation=True,
            max_length=self.seq_len_ans,
            return_tensors="np",
            return_token_type_ids=False,
        )
        ans2 = encoded["input_ids"][0]

        ###########################
        # Process context
        ###########################
        Hn = ast.literal_eval(entry.Hn)
        n_samples = min((len(Hn), self.n_paras))
        contx = sample(Hn, n_samples)

        # Process context
        paras = np.zeros((self.n_paras, self.seq_len_para), dtype=np.int)
        paras_mask = np.zeros((self.n_paras, self.seq_len_para), dtype=np.int)
        for ith, para in enumerate(contx):
            encoded = self.tokenizer(
                para,
                padding="max_length",
                truncation=True,
                max_length=self.seq_len_para,
                return_tensors="np",
                return_token_type_ids=False,
            )
            paras[ith] = encoded["input_ids"]
            paras_mask[ith] = encoded["attention_mask"]

        return {
            "ques": ques,
            "ques_mask": ques_mask,
            "ans1": ans1,
            "ans2": ans2,
            "ans1_mask": ans1_mask,
            "paras": paras,
            "paras_mask": paras_mask,
        }

    def read_datasetfile(self, path_file):
        # NOTE: In future, when data format is Parquet, this line must be fixed
        df = pd.read_csv(path_file, index_col=None, header=0)

        # self.docId          = []
        # self.ques_plain     = []
        self.ques = []
        self.ques_mask = []
        self.ans1 = []
        self.ans2 = []
        self.ans1_mask = []
        self.paras = []
        self.paras_mask = []

        gc.collect()

        ######################
        # Fill self.ques, self.ans1,  self.ans2,
        # answers' mask and index
        ######################
        if self.num_workers > 1:
            entries = ParallelHelper(
                self.f_process_file_multi,
                df,
                lambda dat, l, h: dat.iloc[l:h],
                self.num_workers,
            ).launch()
        else:
            entries = list(map(self.f_process_file_single, df.itertuples()))

        for entry in entries:
            # self.docId.append(entry['docId'])
            # self.ques_plain.append(entry['ques_plain'])
            self.ques.append(entry["ques"])
            self.ques_mask.append(entry["ques_mask"])
            self.ans1.append(entry["ans1"])
            self.ans2.append(entry["ans2"])
            self.ans1_mask.append(entry["ans1_mask"])
            self.paras.append(entry["paras"])
            self.paras_mask.append(entry["paras_mask"])
