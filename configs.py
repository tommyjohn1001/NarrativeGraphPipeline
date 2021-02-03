"""This file processes arguments and configs"""

import argparse
import logging
import sys
import os

import torch


###############################
# Đọc các argument từ CLI
###############################
parser = argparse.ArgumentParser()

parser.add_argument("--working_place", choices=['local', 'remote', 'local2'],
                    help="working environment", default='local')
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--n_cpus", type=int, default=4)
parser.add_argument("--n_epochs", type=int, default=10)
parser.add_argument("--bert_model", type=str, default="bert-base-uncased",
                    help="default pretrain BERT model")
parser.add_argument("--task", type=str, help="Select task to do")
parser.add_argument("--device", type=str, default="default", choices=["default", "cpu", "cuda"],
                    help="Select device to run")

# args = parser.parse_args()
args, _ = parser.parse_known_args()


###############################
# Thêm các argument
###############################
## args = working_place
if args.working_place == "local":
    args.init_path  = "/Users/hoangle/Projects/VinAI"
elif args.working_place == "remote":
    args.init_path  = "/home/ubuntu"
elif args.working_place == "local2":
    args.init_path = "/home/tommy/Projects/VinAI"
else:
    print("Err: Invalid init path. Exiting..")
    sys.exit(1)

## args = bert_model
path = f"{args.init_path}/_pretrained/BERT/{args.bert_model}/"
if os.path.exists(path):
    args.bert_model = path

## args = device
if args.device == "default":
    args.device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    args.device = torch.device(args.device)


###############################
# Cấu hình logging
###############################
logging.basicConfig(format='%(asctime)s: %(message)s', datefmt='%b/%d/%Y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)


###############################
# Cấu hình các biến config
###############################
configs = {
    ## threshold used to select golden passages
    'DELTA'             : 0.1,

    ## maximum number of characters to be tokenized
    'MAX_SEQ_LEN_CH'    : 1024,
    ## maximum number of words to be tokenized
    'MAX_SEQ_LEN_W'     : 512,

    'MULTI_GPUS'        : True if torch.cuda.device_count() > 0 else False
}