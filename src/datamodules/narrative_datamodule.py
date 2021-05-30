from torch.utils.data import DataLoader
import pytorch_lightning as plt

from src.datamodules.utils import NarrativeDataset, CustomSampler

class NarrativeDataModule(plt.LightningDataModule):
    def __init__(self,
        path_data: str,
        path_vocab: str,
        sizes_dataset: dict,
        sizes_shard: dict,
        batch_size: int = 5,
        seq_len_ques: int = 42,
        seq_len_para: int = 122,
        seq_len_ans: int = 42,
        n_paras: int = 30,
        num_workers: int = 4):

        super().__init__()

        self.batch_size     = batch_size
        self.seq_len_ques   = seq_len_ques
        self.seq_len_para   = seq_len_para
        self.seq_len_ans    = seq_len_ans
        self.n_paras        = n_paras
        self.path_data      = path_data
        self.path_vocab     = path_vocab
        self.num_workers    = num_workers
        self.sizes_dataset  = sizes_dataset
        self.sizes_shard    = sizes_shard

        self.data_train = None
        self.data_test  = None
        self.data_valid = None


    def prepare_data(self):
        """Download/preprocess (tokenizer...) if needed and do not touch with self.data_train,
        self.data_test or self.data_valid.
        """
        pass

    def setup(self, stage):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        dataset_args = {
            'path_data'     : self.path_data,
            'path_vocab'    : self.path_vocab,
            'seq_len_ques'  : self.seq_len_ques,
            'seq_len_para'  : self.seq_len_para,
            'seq_len_ans'   : self.seq_len_ans,
            'n_paras'       : self.n_paras,
            'num_worker'    : self.num_workers
        }
        if stage == "fit":
            self.data_train = NarrativeDataset("train", size_dataset=self.sizes_dataset['train'],
                                               size_shard=self.sizes_shard['train'], **dataset_args)
            self.data_valid = NarrativeDataset("valid", size_dataset=self.sizes_dataset['valid'],
                                               size_shard=self.sizes_shard['train'], **dataset_args)
        else:
            self.data_test  = NarrativeDataset("test", size_dataset=self.sizes_dataset['test'],
                                               size_shard=self.sizes_shard['train'], **dataset_args)

    def train_dataloader(self):
        """Return DataLoader for training."""
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            sampler=CustomSampler(self.sizes_dataset['train'])
        )

    def val_dataloader(self):
        """Return DataLoader for validation."""

        return DataLoader(
            dataset=self.data_valid,
            batch_size=self.batch_size,
            sampler=CustomSampler(self.sizes_dataset['valid'])
        )

    def test_dataloader(self):
        """Return DataLoader for test."""

        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            sampler=CustomSampler(self.sizes_dataset['test'])
        )

    def predict_dataloader(self):
        """Return DataLoader for prediction."""

        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            sampler=CustomSampler(self.sizes_dataset['test'])
        )

    def switch_answerability(self):
        self.data_train.switch_answerability()
