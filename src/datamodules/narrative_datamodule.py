from torch.utils.data import DataLoader
import pytorch_lightning as plt

from src.datamodules.dataset import NarrativeDataset

# from src.datamodules.preprocess import Preprocess


class NarrativeDataModule(plt.LightningDataModule):
    def __init__(
        self,
        path_raw_data: str,
        path_processed_contx: str,
        path_data: str,
        path_bert: str,
        batch_size: int = 5,
        len_ques: int = 42,
        len_para_processing: int = 120,
        len_para: int = 500,
        len_ans: int = 42,
        n_paras: int = 3,
        num_workers: int = 4,
    ):

        super().__init__()

        self.batch_size = batch_size
        self.len_ques = len_ques
        self.len_para_processing = len_para_processing
        self.len_para = len_para
        self.len_ans = len_ans
        self.n_paras = n_paras
        self.path_raw_data = path_raw_data
        self.path_processed_contx = path_processed_contx
        self.path_data = path_data
        self.path_bert = path_bert
        self.num_workers = num_workers

        self.data_train = None
        self.data_test = None
        self.data_valid = None

    def prepare_data(self):
        """Download/preprocess (tokenizer...) if needed and do not touch with self.data_train,
        self.data_test or self.data_valid.
        """
        # NOTE: This is temporarily commented for saving time. Uncomment it if needing preprocessing data
        # Preprocess(
        #     num_workers=self.num_workers,
        #     len_para_processing=self.len_para_processing,
        #     n_paras=self.n_paras,
        #     path_raw_data=self.path_raw_data,
        #     path_processed_contx=self.path_processed_contx,
        #     path_data=self.path_data,
        # ).preprocess()
        pass

    def setup(self, stage):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        dataset_args = {
            "path_data": self.path_data,
            "path_bert": self.path_bert,
            "len_ques": self.len_ques,
            "len_para": self.len_para,
            "len_ans": self.len_ans,
            "n_paras": self.n_paras,
            "num_worker": self.num_workers,
        }
        if stage == "fit":
            self.data_train = NarrativeDataset("train", **dataset_args)
            self.data_valid = NarrativeDataset("valid", **dataset_args)
        else:
            self.data_test = NarrativeDataset("test", **dataset_args)

    def train_dataloader(self):
        """Return DataLoader for training."""
        return DataLoader(dataset=self.data_train, batch_size=self.batch_size)

    def val_dataloader(self):
        """Return DataLoader for validation."""

        return DataLoader(dataset=self.data_valid, batch_size=self.batch_size)

    def test_dataloader(self):
        """Return DataLoader for test."""

        return DataLoader(dataset=self.data_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        """Return DataLoader for prediction."""

        return DataLoader(dataset=self.data_test, batch_size=self.batch_size)

    def switch_answerability(self):
        self.data_train.switch_answerability()
