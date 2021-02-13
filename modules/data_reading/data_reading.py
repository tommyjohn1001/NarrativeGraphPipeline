'''This file contains code reading raw data and do some preprocessing'''
from multiprocessing import Pool, Queue
import glob
import re

from bs4 import BeautifulSoup
from tqdm import tqdm

from modules.utils import save_object
from configs import args, logging


def clean_sentence(s:str) -> str:
    """Clean sentence

    Args:
        s (str): sentence to be cleaned

    Returns:
        str: output sentence
    """

    s = re.sub(r"(\n|\t)", "", s)
    s = re.sub(r"\s{2,}", " ", s)
    s = re.sub(r" CUT TO:", "", s)
    s = re.sub(r"^\s", "", s)

    return s.lower()


class DataReading():
    """ This class does the following:
    - Read raw data from files
    - Decompose each document into reasonable paragraphs
    - Do ssome basic prepricessing

    All processed data are backed up.
    """

    def __init__(self) -> None:
        self.path_rawDataFiles  = glob.glob(f"{args.init_path}/_data/QA/NarrativeQA/tmp/*.content")
        self.path_rawParagraphs = f"{args.init_path}/_data/QA/NarrativeQA/backup_folder/raw_paragraphs.pkl"

    def f_trigger(self, path) -> list:
        """This function is used to run in parallel to read and initially preprocess
        raw documents.

        Args:
            path (list): list of paths to raw document

        Returns:
            list: list of paragraphs
        """

        # for path in list_path:
        ##########################################
        ### 1: Read documents from file
        ##########################################
        with open(path, 'r', encoding="ISO-8859-1") as dat_file:
            data    = dat_file.read()

        ##########################################
        ### 2: Classify and decompose documents
        ### into paragraphs
        ##########################################
        if re.search(r"\<html\>", data):
            return {
                'id_document'   : re.findall(r"tmp\/(.+)\.content", path)[0],
                'paragraphs'    : self.read_script(data)
            }

        return {
            'id_document'   : re.findall(r"tmp\/(.+)\.content", path)[0],
            'paragraphs'    : self.read_story(data)
        }


    def trigger(self):
        """ Start reading and processing data
        """

        logging.info("1. Start reading raw data and decompose into paragraphs")

        ##########################################
        ## Step 1: Read and preprocess raw data in parallel
        ##########################################
        logging.info("1.1. Read and preprocess raw data in parallel")

        # list_rawDocuments   = ParallelHelper(self.f_trigger, self.path_rawDataFiles).launch()
        with Pool(args.n_cpus) as p:
            list_rawDocuments   = list(tqdm(p.imap(self.f_trigger, self.path_rawDataFiles),
                                            total=len(self.path_rawDataFiles)))

        ##########################################
        ## Step 2: Store paragraphs
        ##########################################
        logging.info("1.2. Store paragraphs")

        save_object(self.path_rawParagraphs, list_rawDocuments)


    def read_script(self, data:str) -> list:
        """Decompose script into list of paragraphs

        Args:
            data (str): Data read from file

        Returns:
            list: list of paragraphs
        """

        soup = BeautifulSoup(data, 'html.parser')

        def filter_scence(s:str):
            if s is None:
                return True

            condition1  = re.search(r"(?:EXT|INT)", s) is not None
            condition2  = re.search(r"[a-z]", s) is None
            return condition1 and condition2


        paras   = list()
        para    = None

        def add_with_verify(l: list, s: str) -> list:
            if s:
                s = re.sub(r"(\n|\t)", "", s)
                s = re.sub(r"\s{2,}", " ", s)
                s = re.sub(r" CUT TO:", "", s)
                s = re.sub(r"^\s", "", s)

                if len(s) > 1:
                    l.append(s.lower())

        for sentence in soup.pre.find_all('b'):
            text        = sentence.nextSibling.string if sentence.nextSibling else None

            if filter_scence(text):
                continue

            sentence    = sentence.text

            if filter_scence(sentence):
                if para is not None:
                    paras.append(para)

                ## Add scence transition sentence to 'para'
                para = list()
                add_with_verify(para, sentence)

                ## Add context sentence(s) to 'para'
                add_with_verify(para, text)

            elif para is not None:
                add_with_verify(para, sentence)
                add_with_verify(para, text)

        return paras


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
            para = re.sub(r'\n', '', para)

            if len(para) > 0:
                paragraphs.append(para.lower())

        return paragraphs


if __name__ == '__main__':
    data_reader = DataReading()

    data_reader.trigger()
