'''This file contains functions used commonly by entire project'''
import multiprocessing
import logging
import pickle
import gzip
import os

from tqdm import tqdm


########################################################
# Functions to deal with `pkl` files
########################################################
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


def save_object(path: str, obj_file: object, is_dataframe:bool = False) -> object:
    """
    Save object to pickle gzip file.

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
        obj_file.to_pickle(path)
    else:
        with gzip.open(path, 'w+') as dat_file:
            pickle.dump(obj_file, dat_file)


########################################################
# Simple file manipulating operations
########################################################
def check_file_existence(path: str) -> bool:
    """
    Check whether file given by `path` exists.
    Args:
        path (str): path of file to be checked

    Returns:
        bool value stating the existence of file
    """

    return os.path.isfile(path)

class ParallelHelper:
    def __init__(self, f_task: object, data: list, n_cores: int = 4, *args):
        self.n_data = len(data)

        self.queue  = multiprocessing.Queue()
        self.pbar   = tqdm(total=self.n_data)

        self.jobs = list()
        for ith in range(n_cores):
            low_bound = ith * self.n_data // n_cores
            hi_bound = (ith + 1) * self.n_data // n_cores \
                if ith < (n_cores - 1) else self.n_data

            p = multiprocessing.Process(target=f_task,
                                        args=(data[low_bound: hi_bound],
                                              self.queue, *args))
            self.jobs.append(p)

    def launch(self) -> list:
        """
        Launch parallel process
        Returns: a list after running parallel task

        """
        dataset = list()

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
            job.join()


        return dataset
