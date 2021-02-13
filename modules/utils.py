'''This file contains functions used commonly by entire project'''
import logging
import pickle
import gzip
import os


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


def save_object(path, obj_file) -> object:
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
