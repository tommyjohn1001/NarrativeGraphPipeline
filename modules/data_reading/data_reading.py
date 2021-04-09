'''This file contains code reading raw data and do some preprocessing'''
from multiprocessing import Manager
import re, json, gc

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_dataset
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import spacy

from modules.utils import save_object, check_exist, ParallelHelper
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

        return context

    def export_para(self, toks):
        return re.sub(r'( {2,}|\t)', ' ', ' '.join(toks)).strip()

    def process_context_movie(self, context, start, end) -> list:
        """Process context and split into paragrapgs. Dedicated to movie context.

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
        if start_ >= end_:
            start_, end_    = 0, len(context)

        context = context[start_:end_]

        ## Clean context and split into paras
        sentences   = self.clean_context_movie(context).split('\n')

        paras       = np.array([])
        n_tok       = 0 
        para        = []
        for sent in sentences:
            ## Tokenize, remove stopword
            sent_   = [tok.text for tok in nlp(sent)
                        if not (tok.is_stop or tok.is_punct)]

            ## Concat sentece if not exceed paragraph max len
            if n_tok + len(sent_) < 50:
                n_tok   += len(sent_)
                para.extend(sent_)
            else:
                paras   = np.append(paras, self.export_para(para))
                n_tok   = 0
                para    = []


        return paras.tolist()

    def process_context_gutenberg(self, context, start, end):
        """Process context and split into paragrapgs. Dedicated to gutenberg context.

        Args:
            context (str): context (book or movie script)
            start (str): start word of context
            end (str): end word of context

        Returns:
            list: list of paras
        """

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
        if start_ >= end_:
            start_, end_    = 0, len(context)

        context = context[start_:end_]

        ## Clean context and split into paras
        sentences   = self.clean_context_gutenberg(context).split('\n')

        paras       = np.array([])
        n_tok       = 0
        para        = []
        for sent in sentences:
            ## Tokenize, remove stopword
            sent_   = [tok.text for tok in nlp(sent)
                    if not (tok.is_stop or tok.is_punct)]

            if len(sent_) > 50:
                # Long paragraph: split into sentences and
                # apply concatenating strategy

                # Finish concateraning para
                if len(para) > 0:
                    paras = np.append(paras, self.export_para(para))
                    n_tok   = 0
                    para    = []

                # Split into sentences and
                # apply concatenating strategy
                for sub_sent in nlp(sent).sents:
                    sent_   = [tok.text for tok in sub_sent
                            if not (tok.is_stop or tok.is_punct)]

                    if n_tok + len(sent_) < 50:
                        n_tok   += len(sent_)
                        para.extend(sent_)
                    else:
                        paras   = np.append(paras, self.export_para(para))
                        n_tok   = len(sent_)
                        para    = sent_

                if len(para) > 0:
                    paras = np.append(paras, self.export_para(para))
                    n_tok   = 0
                    para    = []

            else:
                if n_tok + len(sent_) < 50:
                    n_tok   += len(sent_)
                    para.extend(sent_)
                else:
                    paras   = np.append(paras, self.export_para(para))
                    n_tok   = len(sent_)
                    para    = sent_

        if para != "":
            paras   = np.append(paras, self.export_para(para))
            n_tok   = len(sent_)
            para    = sent_


        return paras.tolist()


    def f_process_contx(self, keys, queue, args):
        processed_contx = args[0]
        for key in keys:
            context, kind, start, end = processed_contx[key]

            if kind == "movie":
                paras   = self.process_context_movie(context, start, end)
            else:
                paras   = self.process_context_gutenberg(context, start, end)

            queue.put((key, paras))


    def process_ques_ans(self, text):
        """Process question/answers

        Args:
            text (str): question or answers

        Returns:
            str: processed result
        """
        tok = [tok.text for tok in nlp(text)
               if not (tok.is_stop or tok.is_punct)]

        return ' '.join(tok)

    ############################################################
    # Methods to find golden passages given question or answers
    ############################################################
    def find_golden(self, query, wm: list) -> set:

        ## Calculate score of query for each para
        query_  = np.expand_dims(wm[query], 0)
        wm_     = wm[:-3]

        scores  = cosine_similarity(query_, wm_).squeeze(0)


        goldens   = set()

        for ith, score in enumerate(scores):
            if score > 0:
                goldens.add(ith)
                if len(goldens) == args.n_paras:
                    break

        if len(goldens) < args.n_paras:
            for ith, score in enumerate(scores):
                if score == 0:
                    goldens.add(ith)
                    if len(goldens) == args.n_paras:
                        break

        return goldens

    def process_entry(self, entries, queue, args):
        """This function is used to run in parallel to read and initially preprocess
        raw documents. Afterward, it puts a dict containing result into queue.

        Args:
            document (dict): a document of dataset
        """
        processed_contx = args[0]

        for entry in entries:
            #########################
            ## Preprocess context
            #########################
            paras   = processed_contx[entry['document']['id']]


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



            queue.put({
                'doc_id'    : entry['document']['id'],
                'question'  : ' '.join(entry['question']['tokens']),
                'answers'   : [' '.join(x['tokens']) for x in entry['answers']],
                'En'        : [paras[i] for i in golden_paras_answ],
                'Hn'        : [paras[i] for i in golden_paras_ques]
            })


    def trigger_reading_data(self):
        """ Start reading and processing data
        """
        for shard in range(8):
            #########################
            # Load shard of dataset
            #########################
            ### Need to check whether this shard has already been processed
            path    = PATH['dataset_para'].replace("[SPLIT]", self.split).replace("[SHARD]", str(shard))
            if check_exist(path):
                continue
            logging.info(f"= Process dataset: {self.split} - shard {shard}")

            dataset = load_dataset('narrativeqa', split=self.split).shard(8, shard)



            #########################
            # Process contexts first
            #########################
            logging.info(f"= Process context in shard {shard}")

            if check_exist(PATH['processed_contx'].replace("[SPLIT]", self.split)):
                logging.info("=> Backed up processed context file existed. Load it.")
                with open(PATH['processed_contx'].replace("[SPLIT]", self.split), 'r') as d_file:
                    self.processed_contx    = json.load(d_file)


            # Check whether contexts in this shard have been preprocessed
            shared_dict = Manager().dict()  # Prepare for multiprocessing
            for entry in dataset:
                try:
                    id_ = entry['document']['id']
                    _   = self.processed_contx[id_]
                except KeyError:
                    shared_dict[id_] = np.array([entry['document']['text'],
                                                        entry['document']['kind'],
                                                        entry['document']['start'],
                                                        entry['document']['end']])

            # Start processing context in parallel
            list_tuples = ParallelHelper(self.f_process_contx, list(shared_dict.keys()),
                                        lambda d, l, h: d[l:h],
                                        args.num_proc, shared_dict).launch()

            # Update self.processed_contx with already processed contexts
            self.processed_contx.update({it[0]: it[1] for it in list_tuples})

            # Backup processed contexts
            with open(PATH['processed_contx'].replace("[SPLIT]", self.split), 'w+') as d_file:
                json.dump(self.processed_contx, d_file, indent=2, ensure_ascii=False)
                logging.info("=> Backed up processed context.")

            shared_dict = list_tuples = None
            gc.collect()


            #########################
            # Process each entry of dataset
            #########################
            shared_dict = Manager().dict()
            shared_dict.update(self.processed_contx)
            list_documents  = ParallelHelper(self.process_entry, dataset,
                                             lambda d, l, h: d.select(range(l,h)),
                                             args.num_proc, shared_dict).launch()

            save_object(path, pd.DataFrame(list_documents))


if __name__ == '__main__':
    logging.info("* Reading raw data and decompose into paragraphs")

    for splt in ['test']:
        DataReading(splt).trigger_reading_data()
