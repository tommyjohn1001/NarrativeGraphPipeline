'''This file contains code reading raw data and do some preprocessing'''
import multiprocessing
import json
import os
import re

from datasets.arrow_dataset import Dataset
from datasets import load_dataset
from bs4 import BeautifulSoup
from tqdm import tqdm
import pandas as pd

from modules.utils import save_object
from configs import args, logging, PATH


def clean_para(para:str) -> str:
    para = re.sub(r'\n\t', '', para)
    para = re.sub(r'\s{2,}', ' ', para)

    para = para.strip()

    return para.lower()

class DataReading():
    """ This class does the following:
    - Read raw data from files
    - Decompose each document into reasonable paragraphs
    - Do ssome basic prepricessing

    All processed data are backed up.
    """

    def f_trigger(self, documents, queue):
        """This function is used to run in parallel to read and initially preprocess
        raw documents.

        Args:
            path (list): list of paths to raw document

        Returns:
            list: list of paragraphs
        """
        for document in documents:
            ##########################################
            ### Classify and decompose documents into paragraphs
            ##########################################

            ## Remove redundant character '\x00
            text    = document['document']['text'].replace('\x00', '')


            if document['document']['kind'] == 'movie': ## script
                document['document']['text']    = self.read_script(text)
            else:
                document['document']['text']    = self.read_story(text)

            queue.put(document)


    def trigger(self):
        """ Start reading and processing data
        """
        for split in ['train', 'test', 'validation']:
            logging.info("= Preprocess dataset: %s", split)

            dataset = load_dataset('narrativeqa', split=split)

            for nth in range(args.n_shards):
                logging.info(f"= Process shard: {nth}")

                list_documents  = self.process_parallel(self.f_trigger, dataset.shard(args.n_shards, nth))


                logging.info("= Saving dataset: %s", split)

                path    = PATH[f'dataset_para_{split}'].replace("[N_SHARD]", str(nth))
                save_object(path, pd.DataFrame(list_documents), True)


    def process_parallel(self, f_task: object, data: Dataset, n_cores: int = 4) -> list:
        """Parallel processing helper

        Args:
            f_task (object): function to be run by processes
            data (Dataset): dataset to be processed in parallel
            n_cores (int, optional): no. processes. Defaults to 4.

        Returns:
            list: list containing results
        """

        n_data  = len(data)

        queue   = multiprocessing.Queue()
        pbar    = tqdm(total=n_data)

        jobs    = list()
        for ith in range(n_cores):
            low_bound = ith * n_data // n_cores
            hi_bound = (ith + 1) * n_data // n_cores \
                if ith < (n_cores - 1) else n_data

            p = multiprocessing.Process(target=f_task,
                                        args=(data.select(range(low_bound, hi_bound)), queue))
            jobs.append(p)


        dataset = list()

        for job in jobs:
            job.start()

        cnt = 0
        while cnt < n_data:
            while not queue.empty():
                dataset.append(queue.get())
                cnt += 1

                pbar.update()

        for job in jobs:
            job.join()

        pbar.close()
        queue.close()

        return dataset


    def read_script(self, data:str) -> list:
        """Decompose script into list of paragraphs

        Args:
            data (str): Data read from file

        Returns:
            list: list of paragraphs
        """

        ### Parse text
        soup = BeautifulSoup(data, 'html.parser')
        data = ''.join(list(soup.pre.findAll(text=True)))

        paragraphs  = list()
        for para in data.split("\n\n"):
            para    = clean_para(para)

            if len(para) > 0:
                paragraphs.append(para)

        ### Merge 10 every consecutive sentences into a passage
        tmp = list()
        for ith in range(0, len(paragraphs), 5):
            tmp.append(''.join(paragraphs[ith : ith+5]))
        paragraphs  = tmp


        return paragraphs


    def read_story(self, data:str) -> list:
        """Read story document and decompose to paragraphs.
        Due to the format inconsistence of raw data, some redundant texts are included in the final list.

        Args:
            data (str): Raw document to be decompose

        Returns:
            list: list of paragraphs
        """

        paragraphs  = list()
        for para in data.split("\n\n"):
            para    = clean_para(para)

            if len(para) > 0:
                paragraphs.append(para)

        return paragraphs


if __name__ == '__main__':
    logging.info("* Reading raw data and decompose into paragraphs")

    data_reader = DataReading()

    data_reader.trigger()
