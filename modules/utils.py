'''This file contains functions used commonly by entire project'''
import multiprocessing, logging, pickle, gzip, os

import torch.nn as torch_nn
import torch
from tqdm import tqdm

from configs import args


def load_object(path) -> object:
    """
    Load object from pickle gzip file
    :param path: path to file needed to load
    :type path: str
    :return: object stored in file, None if file not existed
    :rtype: object
    """

    if not os.path.isfile(path):
        logging.error(f"Path {path} not found.")

    with gzip.open(path, 'r') as dat_file:
        return pickle.load(dat_file)


def save_object(path: str, obj_file: object, is_dataframe:bool = True) -> object:
    """
    Save object to pickle or csv file.

    Args:
        path (str): path to file needed to store
        obj_file (object): object needed to store

    Returns: object
    None if object is None
    """
    if obj_file is None:
        return None

    if os.path.isfile(path):
        logging.warning("=> File %s will be overwritten.", path)
    else:
        try:
            os.makedirs(os.path.dirname(path))
        except FileExistsError:
            logging.warning("=> Folder %s exists.", str(os.path.dirname(path)))

    if is_dataframe:
        obj_file.to_csv(path, index=False)
    else:
        with gzip.open(path, 'w+') as dat_file:
            pickle.dump(obj_file, dat_file)


def check_exist(path: str) -> bool:
    """
    Check whether file given by `path` exists.
    Args:
        path (str): path of file to be checked

    Returns:
        bool value stating the existence of file
    """

    return os.path.isfile(path)


class ParallelHelper:
    def __init__(self, f_task: object, data: list,
                 data_allocation: object, num_proc: int = 4, *args):
        self.n_data = len(data)

        self.queue  = multiprocessing.Queue()
        self.pbar   = tqdm(total=self.n_data)

        self.jobs = list()
        for ith in range(num_proc):
            lo_bound = ith * self.n_data // num_proc
            hi_bound = (ith + 1) * self.n_data // num_proc \
                if ith < (num_proc - 1) else self.n_data

            p = multiprocessing.Process(target=f_task,
                                        args=(data_allocation(data, lo_bound, hi_bound),
                                              self.queue,
                                              args))
            self.jobs.append(p)

    def launch(self) -> list:
        """
        Launch parallel process
        Returns: a list after running parallel task

        """
        dataset = []

        for job in self.jobs:
            job.start()

        cnt = 0
        while cnt < self.n_data:
            while not self.queue.empty():
                dataset.append(self.queue.get())
                cnt += 1

                self.pbar.update()

        self.pbar.close()

        for job in self.jobs:
            job.terminate()

        for job in self.jobs:
            job.join()


        return dataset

########################################################
# Common layers and functions regarding deep model
########################################################
def transpose(X:torch.Tensor, d1=1, d2=2):
    """Dedicated function to transpose 2 last dimensions of 3D tensor.

    Args:
        X (torch.Tensor): Tensor to be transposed
        d1 (int, optional): First dim. Defaults to 1.
        d2 (int, optional): Second dim. Defaults to 2.

    Returns:
        tensor: Tensor after transposing
    """
    return X.transpose(d1, d2)

class NonLinear(torch_nn.Module):
    def __init__(self, d_in, d_out, activation="sigmoid", dropout=args.dropout):
        super().__init__()

        self.linear     = torch_nn.Linear(d_in, d_out)
        self.activation = torch_nn.GELU()
        self.dropout    = torch_nn.Dropout(dropout)

    def forward(self, X):
        X   = self.activation(self.linear(X))
        X   = self.dropout(X)

        return X
