from glob import glob
import sys
import os


from transformers import BertModel, AdamW, get_linear_schedule_with_warmup
from datasets import load_dataset, Dataset
import torch.nn.functional as torch_f
import torch.nn as torch_nn
import datasets
import torch

from modules.paras_selection.ParasSelection import ParasScoring
from modules.paras_selection.utils import create_tensors
from configs import args, logging, PATH

datasets.logging.set_verbosity_warning()


def train(dataset_train: Dataset, model, optimizer, loss, scheduler=None):
    model.train()

    nth_batch   = 0
    ave_loss    = 0

    for entry in dataset_train:
        src         = list(map(torch.LongTensor, entry['src']))
        attn_mask   = list(map(torch.LongTensor, entry['attn_mask']))
        trg         = list(map(torch.FloatTensor, entry['trg']))

        for ith in range(0, len(src), args.batch_size):
            src_        = torch.vstack(src[ith:ith+args.batch_size]).to(args.device)
            attn_mask_  = torch.vstack(attn_mask[ith:ith+args.batch_size]).to(args.device)
            trg_        = torch.vstack(trg[ith:ith+args.batch_size]).to(args.device)

            ### Push tensors to model to train
            predict     = model(src_, attn_mask_)

            ### Calculate loss and do some stuffs
            output      = loss(predict, trg_)

            output.backward()
            optimizer.step()        

            ### Collect and log some info
            print(f"= Batch {nth_batch:3d}: Loss: {output:9.6f}")
            nth_batch   += 1
            ave_loss    += output

        torch.cuda.empty_cache()


    return ave_loss / nth_batch


def eval(dataset_test: Dataset, model, loss):
    model.eval()

    nth_batch   = 0
    ave_loss    = 0

    with torch.no_grad():
        for entry in dataset_test:
            src         = list(map(torch.LongTensor, entry['src']))
            attn_mask   = list(map(torch.LongTensor, entry['attn_mask']))
            trg         = list(map(torch.FloatTensor, entry['trg']))

            for ith in range(0, len(src), args.batch_size):
                src_        = torch.vstack(src[ith:ith+args.batch_size]).to(args.device)
                attn_mask_  = torch.vstack(attn_mask[ith:ith+args.batch_size]).to(args.device)
                trg_        = torch.vstack(trg[ith:ith+args.batch_size]).to(args.device)

                ### Push tensors to model to train
                predict     = model(src_, attn_mask_)

                ### Calculate loss and do some stuffs
                output      = loss(predict, trg_)

                ### Collect and log some info
                print(f"= Batch {nth_batch:3d}: Loss: {output:9.6f}")
                nth_batch   += 1
                ave_loss    += output

            torch.cuda.empty_cache()


    return ave_loss / nth_batch

if __name__ == '__main__':
    logging.info("")
    paths           = glob("./backup/processed_data/train/data_*.pkl", recursive=True)
    dataset_train   = load_dataset('pandas', data_files=paths)['train']

    paths           = glob("./backup/processed_data/test/data_*.pkl", recursive=True)
    dataset_test    = load_dataset('pandas', data_files=paths)['train']

    dataset_train   = dataset_train.map(create_tensors, 
                                        remove_columns=dataset_train.column_names)
    dataset_test    = dataset_train.map(create_tensors,
                                        remove_columns=dataset_train.column_names)


    model       = ParasScoring().to(args.device)
    optimizer   = AdamW(model.parameters())
    # scheduler   = get_linear_schedule_with_warmup(optimizer, 1000, dataset_train)
    loss        = torch_nn.MSELoss()


    ## Train
    for n_epoch in range(2):

        loss_train  = train(dataset_train, model, optimizer, loss)

        loss_test   = eval(dataset_test, model, loss)


        print(f"epoch {n_epoch:2d} | train: loss: {loss_train:7.3f}")

