'''This file contains code reading raw data and do some preprocessing'''
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from datasets import load_dataset
from bs4 import BeautifulSoup
from scipy import spatial
import pandas as pd
import numpy as np
import spacy

from modules.utils import save_object, ParallelHelper, check_file_existence
from configs import args, logging, PATH


log = logging.getLogger("spacy")
log.setLevel(logging.ERROR)

nlp             = spacy.load("en_core_web_sm")
nlp.add_pipe('sentencizer')


class DataReading():
    def __init__(self, split) -> None:
        self.processed_contx    = {}
        self.split              = split

    ############################################################
    # Methods to process context, question and answers
    ############################################################
    def clean_end(self, text:str):
        text = text.split(' ')
        return ' '.join(text[:-2])

    def clean_context_movie(self, context:str) -> str:
        context = context.lower().strip()

        context = re.sub(r'( {2,}|\t)', ' ', context)
        context = re.sub(r' \n ', '\n', context)
        # context = re.sub(r'\n ', ' ', context)
        context = re.sub(r"(?<=\w|,)\n(?=\w| )", ' ', context)
        context = re.sub(r'\n{2,}', '\n', context)

        return context

    def clean_context_gutenberg(self, context:str) -> str:
        context = context.lower().strip()

        context = re.sub(r"(?<=\w|,|;|-)\n(?=\w| )", ' ', context)
        context = re.sub(r'( {2,}|\t)', ' ', context)
        # context = re.sub(r' \n ', '\n', context)
        # context = re.sub(r'\n ', ' ', context)
        context = re.sub(r'\n{2,}', '\n', context)

    def process_context_movie(self, context, start, end) -> list:
        """Process context and split into paragrapgs

        Args:
            context (str): context (book or movie script)
            start (str): start word of context
            end (str): end word of context

        Returns:
            list: list of paras
        """

        ## Extract text from HTML
        soup    = BeautifulSoup(context, 'html.parser')
        if soup.pre is not None:
            context = ''.join(list(soup.pre.findAll(text=True)))

        ## Use field 'start' and 'end' provided
        start_  = start.lower()
        end_    = end.lower()
        end_    = self.clean_end(end_)

        start_  = context.find(start_)
        end_    = context.rfind(end_)
        if start_ == -1:
            start_ = 0
        if end_ == -1:
            end_ = len(context)

        context = context[start_:end_]

        ## Clean context and split into paras
        sentences   = self.clean_context_movie(context).split('\n')

        paras       = []
        n_tok       = 0 
        para        = ""
        for sent in sentences:
            ## Tokenize, remove stopword and lemmatization
            sent_   = [tok.lemma_ for tok in nlp(sent)
                       if not (tok.is_stop or tok.is_punct)]

            ## Concat sentece if not exceed paragraph max len
            if n_tok + len(sent_) < 50:
                n_tok   += len(sent_)
                para    = para + ' ' + ' '.join(sent_)
            else:
                paras.append(para)
                n_tok   = 0
                para    = ""


        return paras




        return context

    def process_context_gutenberg(self, context, start, end):
        soup    = BeautifulSoup(context, 'html.parser')
        if soup.pre is not None:
            context = ''.join(list(soup.pre.findAll(text=True)))

        # Use field 'start' and 'end' provided
        start_  = start.lower()
        end_    = end.lower()
        end_    = self.clean_end(end_)

        start_  = context.find(start_)
        end_    = context.rfind(end_)
        if start_ == -1:
            start_ = 0
        if end_ == -1:
            end_ = len(context)

        context = context[start_:end_]

        ## Clean context and split into paras
        sentences   = self.clean_context_gutenberg(context).split('\n')

        paras       = np.array([])
        n_tok       = 0
        para        = ""
        for sent in sentences:
            ## Tokenize, remove stopword
            sent_   = [tok.text for tok in nlp(sent)
                    if not (tok.is_stop or tok.is_punct)]

            if len(sent_) > 50:
                # Long paragraph: split into sentences and
                # apply concatenating strategy

                # Finish concateraning para
                if para != "":
                    paras = np.append(paras, para)
                    n_tok   = 0
                    para    = ""

                # Split into sentences and
                # apply concatenating strategy
                for sub_sent in nlp(sent).sents:
                    sent_   = [tok.text for tok in sub_sent
                            if not (tok.is_stop or tok.is_punct)]

                    if n_tok + len(sent_) < 50:
                        n_tok   += len(sent_)
                        if para != "":
                            para    = para + ' ' + ' '.join(sent_)
                        else:
                            para    = ' '.join(sent_)
                    else:
                        paras   = np.append(paras, para)
                        n_tok   = len(sent_)
                        para    = ' '.join(sent_)

                if para != "":
                    paras = np.append(paras, para)
                    n_tok   = 0
                    para    = ""

            else:
                if n_tok + len(sent_) < 50:
                    n_tok   += len(sent_)
                    if para != "":
                        para    = para + ' ' + ' '.join(sent_)
                    else:
                        para    = ' '.join(sent_)
                else:
                    paras = np.append(paras, para)
                    n_tok   = 0
                    para    = ""

        if para != "":
            paras   = np.append(paras, para)
            n_tok   = len(sent_)
            para    = ' '.join(sent_)

        paras = paras.tolist()

        return paras


    def process_contx(self, keys, queue):
        for key in keys:
            context, kind, start, end = self.processed_contx[key]

            if kind == "movie":
                paras   = self.process_context_movie(context, start, end)
            else:
                paras   = self.process_context_gutenberg(context, start, end)

            self.processed_contx[key]   = paras

            queue.put(1)


    def process_ques_ans(self, text):
        """Process question/answers

        Args:
            text (str): question or answers

        Returns:
            str: processed result
        """
        tok = [tok.lemma_ for tok in nlp(text)
               if not (tok.is_stop or tok.is_punct)]

        return ' '.join(tok)

    ############################################################
    # Methods to find golden passages given question or answers
    ############################################################
    def get_score(self, v1, v2):
        if np.linalg.norm(v1) * np.linalg.norm(v2) == 0:
            return -1

        return 1 - spatial.distance.cosine(v1, v2)

    def find_golden(self, query, wm: list) -> set:

        ## Calculate score of query for each para
        scores = [
            (ith, self.get_score(wm[query], wm[ith]))
            for ith in range(0, len(wm) - 3)
        ]

        goldens   = set()

        for score in scores:
            if score[1] > 0:
                goldens.add(score[0])
                if len(goldens) == args.n_paras:
                    break

        if len(goldens) < args.n_paras:
            for score in scores:
                if score[1] == 0:
                    goldens.add(score[0])
                    if len(goldens) == args.n_paras:
                        break

        return goldens

    def process_entry(self, entry):
        """This function is used to run in parallel to read and initially preprocess
        raw documents. Afterward, it puts a dict containing result into queue.

        Args:
            document (dict): a document of dataset
        """
        #########################
        ## Preprocess context
        #########################
        paras   = self.processed_contx[entry['document']['id']]


        #########################
        ## Preprocess question and answer
        #########################
        ques    = self.process_ques_ans(entry['question']['text'])
        ans1    = self.process_ques_ans(entry['answers'][0]['text'])
        answ2   = self.process_ques_ans(entry['answers'][1]['text'])


        #########################
        ## TfIdf vectorize
        #########################
        tfidfvectorizer = TfidfVectorizer(analyzer='word', stop_words='english',
                                            ngram_range=(1, 3), max_features=500000)

        wm = tfidfvectorizer.fit_transform(paras + [ques, ans1, answ2]).toarray()

        question, answer1, answer2 = len(wm) - 3, len(wm) - 2, len(wm) - 1


        #########################
        ## Find golden paragraphs from
        ## question and answers
        #########################
        golden_paras_ques   = self.find_golden(question, wm)

        golden_paras_answ1  = self.find_golden(answer1, wm)
        golden_paras_answ2  = self.find_golden(answer2, wm)
        golden_paras_answ   = golden_paras_answ1 | golden_paras_answ2



        return {
            'doc_id'    : entry['document']['id'],
            'question'  : ' '.join(entry['question']['tokens']),
            'answers'   : [' '.join(x['tokens']) for x in entry['answers']],
            'En'        : [paras[i] for i in golden_paras_answ],
            'Hn'        : [paras[i] for i in golden_paras_ques]
        }


    def trigger_reading_data(self):
        """ Start reading and processing data
        """
        #########################
        # Process contexts first
        #########################
        for entry in load_dataset('narrativeqa', split=self.split):
            try:
                id_ = entry['document']['id']
                _   = self.processed_contx[id_]
            except KeyError:
                self.processed_contx[id_] = np.array([entry['document']['text'],
                                                      entry['document']['kind'],
                                                      entry['document']['start'],
                                                      entry['document']['end']])

        ParallelHelper(self.f_process_text, self.processed_contx.keys(),
                       lambda d, lo, hi: d[lo, hi],
                       num_proc=args.num_proc).launch()


        #########################
        # Process each entry of dataset
        #########################
        for shard in range(8):
            ### Need to check whether this shard has already been processed
            path    = PATH['dataset_para'].replace("[SPLIT]", self.split).replace("[SHARD]", str(shard))
            if check_file_existence(path):
                continue


            logging.info(f"= Process dataset: {self.split} - shard {shard}")

            dataset = load_dataset('narrativeqa', split=self.split).shard(8, shard)

            list_documents = [self.process_entry(entry) for entry in dataset]

            save_object(path, pd.DataFrame(list_documents))


if __name__ == '__main__':
    logging.info("* Reading raw data and decompose into paragraphs")

    for split in ['validation', 'train', 'test']:
        DataReading(split).trigger_reading_data()
