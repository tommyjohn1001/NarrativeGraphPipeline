from glob import glob
import gc

import pandas as pd

from modules.paras_selection.ParasSelection import ParasSelection
from modules.paras_selection.utils import conv

if __name__ == '__main__':
    # paras_selection = ParasSelection()
    # paras_selection.trigger_inference()

#     for split in ["train", "test", "valid"]:
#         path_  = f"backup/processed_data/{split}_/data_*.csv"
#         path_  = glob(path_)
#         path_.sort()

#         for ith, path in enumerate(path_):
#             print(path)

#             df = pd.read_csv(path).drop(['Unnamed: 0'], axis=1)
            
#             df1 = df.iloc[:len(df)//2]
#             df2 = df.iloc[len(df)//2:]

#             df1.to_csv(f"backup/processed_data/{split}/data_{ith}_1.csv",
#                     index=False)
#             df2.to_csv(f"backup/processed_data/{split}/data_{ith}_2.csv",
#                     index=False)

#             df = None
#             gc.collect()


    conv()