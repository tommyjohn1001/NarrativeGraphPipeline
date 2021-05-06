"""This file processes arguments and configs"""

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

# args = parser.parse_args()
args, _ = parser.parse_known_args()


###############################
# Add several necessary argument
###############################


## args = device
args.device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


## other args
args.multi_gpus = torch.cuda.device_count() > 0

path_bert   = "/root/BERT/bert-base-uncased"
if os.path.isdir(path_bert):
    args.bert_model         = path_bert
else:
    args.bert_model         = "bert-base-uncased"

args.seq_len_ques       = 40
args.seq_len_dataprocess= 50
args.seq_len_para       = 52        # max len of each para of context after DataReading
args.seq_len_ans        = 40        # maximum answer length of dataset
args.n_paras            = 40
args.d_embd             = 200
args.d_hid              = 256
args.max_len_ans        = 12        # maximum inferring steps of decoder
args.min_count_PGD      = 10        # min occurences of word to be added to vocab of PointerGeneratorDecoder
args.d_vocab            = 27156     # Vocab size
args.dropout            = 0.2
args.n_layers           = 5

args.graph_d_project    = 2048
args.n_edges            = 3120

args.max_breadth        = 8


###############################
# Config path of backup files
###############################
PATH    = {
    ## Paths associated with Data Reading
    'dataset_para'      : "backup/[SPLIT]/data_[SHARD].csv",
    'processed_contx'   : "backup/proc_contx_[SPLIT].json",
    'vocab'             : "backup/vocab.txt",
    'saved_model'       : "backup/model.pt",
    'saved_chkpoint'    : "backup/chkpoint.pth.tar",
    'prediction'        : "backup/predictions.json",
    'log'               : "run.log"
}


###############################
# Config logging
###############################
logging.basicConfig(filename=PATH['log'], filemode='a+', format='%(asctime)s: %(message)s', datefmt='%b-%d-%Y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)
