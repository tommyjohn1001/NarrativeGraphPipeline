
import os

from torch.utils.data import Dataset
import pandas as pd,\
    numpy as np,\
    torch


class SimpleDataset(Dataset):
    def __init__(self, split, path_data: str = None) -> None:
        super().__init__()

        path    = path_data.replace("[SPLIT]", split)
        assert os.path.isfile(path), f"File {path} not existed."

        self.data   = pd.read_parquet(path)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index   = index.to_list()

        entry = self.data.iloc[index]
        return {
            'sent_ids'  : np.array(entry['sent_ids'], copy=True),
            'sent_masks': np.array(entry['sent_masks'], copy=True)
        }
        