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
import unidecode

from modules.utils import save_object, check_exist, ParallelHelper
from configs import args, logging, PATH


log = logging.getLogger("spacy")
log.setLevel(logging.ERROR)

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe('sentencizer')


class DataReading():
    def __init__(self, split) -> None:
        self.processed_contx    = {}
        self.split              = split

    ############################################################
    # Methods to process context, question and answers
    ############################################################
    def clean_context_movie(self, context:str) -> str:
        context = context.lower().strip()

        context = re.sub(r"(style\=\".*\"\>|class\=.*\>)", '', context)

        context = re.sub(r' \n ', '\n', context)
        # context = re.sub(r'\n ', ' ', context)
        context = re.sub(r"(?<=\w|,)\n(?=\w| )", ' ', context)
        context = re.sub(r'\n{2,}', '\n', context)
        context = re.sub(r'( {2,}|\t)', ' ', context)

        return context

    def clean_context_gutenberg(self, context:str) -> str:
        context = context.lower().strip()

        context = re.sub(r"(style\=\".*\"\>|class\=.*\>)", '', context)

        context = re.sub(r"(?<=\w|,|;|-)\n(?=\w| )", ' ', context)
        context = re.sub(r'( {2,}|\t)', ' ', context)
        # context = re.sub(r' \n ', '\n', context)
        # context = re.sub(r'\n ', ' ', context)
        context = re.sub(r'\n{2,}', '\n', context)


        return context

    def clean_sent(self, sent):
        sent    = re.sub(r'( {2,}|\t)', ' ', sent)
        sent    = re.sub(r'(\*|\{|\}|\=|\-|\`|\"|\\){2,}', '', sent)
        sent    = sent.strip()

        return sent

    def extract_para(self, toks):
        return re.sub(r'( {2,}|\t)', ' ', ' '.join(toks)).strip()

    def extract_html(self, context):
        soup    = BeautifulSoup(context, 'html.parser')
        context = unidecode.unidecode(soup.text)
        return  context


    def f_process_contx(self, keys, queue, arg):
        processed_contx = arg[0]

        for key in keys:
            context, kind   = processed_contx[key]

            ###############################
            ## Clean context and break
            ## into separated sentences
            ###############################
            ## Breaking method use 2 methods:
            ##  - Breaking by character '\n
            ##  - Breaking by spacy's sentencizer

            ## Extract text from HTML
            context = self.extract_html(context)

            ## Clean context and use 1st method to break into sentences
            if kind == "movie":
                sentences   = self.clean_context_movie(context).split('\n')
            else:
                sentences   = self.clean_context_gutenberg(context).split('\n')

            ## Use 2nd method to break into sentences
            tmp     = []
            for sent in sentences:
                sent    = self.clean_sent(sent)
                sent_   = [tok.text for tok in nlp(sent)]

                if len(sent_) < args.seq_len_dataprocess:
                    tmp.append(sent_)
                else:
                    for sub_sent in nlp(sent).sents:
                        sent_   = [tok.text for tok in sub_sent]
                        tmp.append(sent_)

            sentences   = tmp


            ###############################
            ## Construc paragraphs from sentencces
            ###############################
            paras       = np.array([])
            para        = []
            para_n_toks = 0
            for sent_toks in sentences:
                if para_n_toks + len(sent_toks) > args.seq_len_dataprocess:
                    if para_n_toks > 0:
                        paras   = np.append(paras, self.extract_para(para))
                    para        = sent_toks
                    para_n_toks = len(sent_toks)
                else:
                    para.extend(sent_toks)
                    para_n_toks += len(sent_toks)
            if para_n_toks:
                paras   = np.append(paras, self.extract_para(para))


            queue.put((key, paras))


    def process_ques_ans(self, text):
        """Process question/answers

        Args:
            text (str): question or answers

        Returns:
            str: processed result
        """
        tok = [tok.text for tok in nlp(text)]

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

    def process_entry(self, entries, queue, arg):
        """This function is used to run in parallel to read and initially preprocess
        raw documents. Afterward, it puts a dict containing result into queue.

        Args:
            document (dict): a document of dataset
        """
        processed_contx = arg[0]

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
                                                 entry['document']['kind']])

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
