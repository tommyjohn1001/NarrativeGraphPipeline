"""This file processes arguments and configs"""

import argparse
import logging

import torch


###############################
# Read arguments from CLI
###############################
parser = argparse.ArgumentParser()

parser.add_argument("--batch", type=int, default=1)
parser.add_argument("--num_proc", type=int, default=4, help="number of processes")
parser.add_argument("--n_epochs", type=int, default=10)
parser.add_argument("--bert_model", type=str, default="bert-base-uncased",
                    help="default pretrain BERT model")
parser.add_argument("--n_shards", type=int, help="Number of chunks to split from large dataset",
                    default=8)
parser.add_argument("--device", type=str, default="default", choices=["default", "cpu", "cuda"],
                    help="Select device to run")
parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate")
parser.add_argument("--w_decay", type=float, default=0, help="Weight decay")

# args = parser.parse_args()
args, _ = parser.parse_known_args()


###############################
# Add several necessary argument
###############################


## args = device
if args.device == "default":
    args.device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    args.device = torch.device(args.device)

## other args
args.multi_gpus = torch.cuda.device_count() > 0

args.seq_len_ques       = 40
args.seq_len_para       = 40
args.seq_len_ans        = 12
args.n_paras            = 40
args.d_embd             = 200
args.d_hid              = 256
args.max_len_ans        = 12
args.teacher_forcing    = 0.3
args.min_count_PGD      = 10        # min occurences of word to be added to vocab of PointerGeneratorDecoder
args.d_vocab            = 124555    # Vocab size
args.dropout            = 0.2
args.n_layers           = 5


###############################
# Config logging
###############################
logging.basicConfig(format='%(asctime)s: %(message)s', datefmt='%b-%d-%Y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)

###############################
# Config path of backup files
###############################
PATH    = {
    ## Paths associated with Data Reading
    'dataset_para'  : "./backup/[SPLIT]/data_[SHARD].csv",
    'vocab_PGD'     : "./backup/vocab_PGD.txt"
}
