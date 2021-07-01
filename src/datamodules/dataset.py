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
        len_ques: int = 42,
        len_para: int = 500,
        len_ans: int = 15,
        n_paras: int = 3,
        num_worker: int = 1,
    ):

        self.len_ques = len_ques
        self.len_para = len_para
        self.len_ans = len_ans
        self.n_paras = n_paras
        self.num_workers = num_worker

        self.tokenizer = BertTokenizer.from_pretrained(path_bert)

        self.curent_ith_file = -1

        self.ques_ids = None
        self.ques_mask = None
        self.ans1_ids = None
        self.ans2_ids = None
        self.ans1_mask = None
        self.context_ids = None
        self.context_mask = None

        self.read_datasetfile(path_data.replace("[SPLIT]", split))

    def __len__(self) -> int:
        return len(self.ques_ids)

    def __getitem__(self, indx):
        if torch.is_tensor(indx):
            indx = indx.tolist()

        return {
            # 'docId'         : self.docId[indx],
            # 'ques_plain'    : self.ques_plain[indx],
            "ques_ids": self.ques_ids[indx],
            "ques_mask": self.ques_mask[indx],
            "ans1_ids": self.ans1_ids[indx],
            "ans2_ids": self.ans2_ids[indx],
            "ans1_mask": self.ans1_mask[indx],
            "context_ids": self.context_ids[indx],
            "context_mask": self.context_mask[indx],
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
            max_length=self.len_ques,
            truncation=True,
            return_tensors="np",
            return_token_type_ids=False,
        )
        ques_ids = encoded["input_ids"][0]
        ques_mask = encoded["attention_mask"][0]

        ###########################
        # Process answers
        ###########################
        ans1, ans2 = entry.answers

        # This trick ensures training process occurs in longer answer
        if len(" ".split(ans1)) < len(" ".split(ans2)):
            ans1, ans2 = ans2, ans1

        encoded = self.tokenizer(
            ans1,
            padding="max_length",
            truncation=True,
            max_length=self.len_ans,
            return_tensors="np",
            return_token_type_ids=False,
        )
        ans1_ids = encoded["input_ids"][0]
        ans1_mask = encoded["attention_mask"][0]

        encoded = self.tokenizer(
            ans2,
            padding="max_length",
            truncation=True,
            max_length=self.len_ans,
            return_tensors="np",
            return_token_type_ids=False,
        )
        ans2_ids = encoded["input_ids"][0]

        ###########################
        # Process context
        ###########################
        contx = entry.context.tolist()

        # Process context
        context_ids = np.zeros((self.n_paras, self.len_para), dtype=np.int)
        context_mask = np.zeros((self.n_paras, self.len_para), dtype=np.int)
        for ith, para in enumerate(contx):
            encoded = self.tokenizer(
                para,
                padding="max_length",
                truncation=True,
                max_length=self.len_para,
                return_tensors="np",
                return_token_type_ids=False,
            )
            context_ids[ith] = encoded["input_ids"]
            context_mask[ith] = encoded["attention_mask"]

        return {
            "ques_ids": ques_ids,
            "ques_mask": ques_mask,
            "ans1_ids": ans1_ids,
            "ans2_ids": ans2_ids,
            "ans1_mask": ans1_mask,
            "context_ids": context_ids,
            "context_mask": context_mask,
        }

    def read_datasetfile(self, path_file):
        df = pd.read_parquet(path_file)

        # self.docId          = []
        # self.ques_plain     = []
        self.ques_ids = []
        self.ques_mask = []
        self.ans1_ids = []
        self.ans2_ids = []
        self.ans1_mask = []
        self.context_ids = []
        self.context_mask = []

        # gc.collect()

        ######################
        # Fill self.ques_ids, self.ans1_ids,  self.ans2_ids,
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
            self.ques_ids.append(entry["ques_ids"])
            self.ques_mask.append(entry["ques_mask"])
            self.ans1_ids.append(entry["ans1_ids"])
            self.ans2_ids.append(entry["ans2_ids"])
            self.ans1_mask.append(entry["ans1_mask"])
            self.context_ids.append(entry["context_ids"])
            self.context_mask.append(entry["context_mask"])
