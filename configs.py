"""This file processes arguments and configs"""

from itertools import combinations
import argparse, logging, os

import torch


###############################
# Read arguments from CLI
###############################
parser = argparse.ArgumentParser()

parser.add_argument("--batch", type=int, default=5)
parser.add_argument("--num_proc", type=int, default=4, help="number of processes")
parser.add_argument("--n_epochs", type=int, default=10)
parser.add_argument("--n_shards", type=int, help="Number of chunks to split from large dataset",
                    default=8)
parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
parser.add_argument("--w_decay", type=float, default=0, help="Weight decay")
parser.add_argument("--is_debug", type=bool, default=False, help="true | false")
parser.add_argument("--task", type=str, default="train", help="train | infer")

# args = parser.parse_args()
args, _ = parser.parse_known_args()


###############################
# Add several necessary argument
###############################


## args = device
args.device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


## other args
args.multi_gpus = torch.cuda.device_count() > 0

args.seq_len_ques   = 40  + 2   # The reason why sequence length of ques, contx and ans
args.seq_len_para   = 120 + 2   # plus 2 is for CLS and SEP token
args.seq_len_ans    = 40  + 2   # maximum answer length of dataset
args.n_paras        = 30
args.d_embd         = 200
args.d_hid          = 64
args.max_len_ans    = 12        # maximum inferring steps of decoder
args.min_count_PGD  = 10        # min occurences of word to be added to vocab of PointerGeneratorDecoder
args.d_vocab        = 32715     # Vocab size
args.dropout        = 0.15
args.n_layers       = 5

args.trans_nheads   = 4
args.trans_nlayers  = 3
args.d_graph        = 2048
args.n_edges        = 3120
args.n_nodes        = len(list(combinations(range(args.n_paras), 2)))

args.beam_size      = 20
args.n_gram_beam    = 5


###############################
# Config path of backup files
###############################
PATH    = {
    'raw_data_dir'      : [
        "/Users/hoangle/Projects/VinAI/_data/NarrativeQA",
        "/home/tommy/Projects/_data/NarrativeQA",
        "/home/ubuntu/NarrativeQA",
        "/root/NarrativeQA"
    ],
    'bert'              : [
        "/root/bert-base-uncased",
        "/home/ubuntu/bert-base-uncased",
        "/Users/hoangle/Projects/VinAI/_pretrained/bert-base-uncased",
        "/home/tommy/Projects/_pretrained/BERT/bert-base-uncased"
    ],
    'glove_embd'        : [
        ".vector_cache/",
        "/Users/hoangle/Projects/VinAI/_pretrained/bert-base-uncased",
        "/home/tommy/Projects/_pretrained/GloVe"
    ],

    ## Paths associated with Data Reading
    'processed_contx'   : "backup/proc_contx/[ID].json",
    'dataset'           : "backup/[SPLIT]/data_[SHARD].csv",
    'vocab'             : "backup/vocab.txt",

    ## Paths associated with models
    'saved_model'       : "backup/model.pt",
    'saved_chkpoint'    : "backup/chkpoint.pth.tar",
    'prediction'        : "backup/predictions.json",
    'memory'            : "backup/memory.pt",

    'log'               : "run.log"
}

for key, val in PATH.items():
    if isinstance(val, list):
        is_found = False
        for path in val:
            if os.path.isdir(path):
                is_found  = True
                PATH[key] = path
                break
        if not is_found:
            raise FileNotFoundError(f"{key} not found")

###############################
# Config logging
###############################
if args.is_debug:
    logging.basicConfig(format='%(asctime)s: %(message)s', datefmt='%b-%d-%Y %H:%M:%S')
else:
    logging.basicConfig(filename=PATH['log'], filemode='a+', format='%(asctime)s: %(message)s', datefmt='%b-%d-%Y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)
