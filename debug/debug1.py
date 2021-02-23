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


def train(dataset: Dataset, model, optimizer, loss, scheduler):
    model.train()

    accum_loss  = 0
    for nth_batch in range(args.batch_size):
        logging.info("Start batch by processing file")

        shard   = dataset.shard(args.batch_size, nth_batch)
        shard   = shard.map(create_tensors, num_proc=args.num_proc, remove_columns=dataset.column_names)


        logging.info("Start training")
        for entry in shard:
            ### Create data tensors from processed dataset
            src         = torch.tensor(entry['src']).to(args.device)
            attn_mask   = torch.tensor(entry['attn_mask']).to(args.device)
            trg         = torch.tensor(entry['trg']).to(args.device)

            ### Push tensors to model to train
            predict     = model(src, attn_mask)

            ### Calculate loss and do some stuffs
            output      = loss(predict, trg)

            output.backward()
            optimizer.step()
            scheduler.step()

            accum_loss  += output

            print(f"First entry: train: loss: {accum_loss:7.3f}")

            sys.exit()

    return accum_loss / dataset.num_rows


if __name__ == '__main__':
    paths           = glob("./backup/processed_data/valid/data_0.pkl")
    dataset_train   = load_dataset('pandas', data_files=paths)['train']


    model       = ParasScoring().to(args.device)
    optimizer   = AdamW(model.parameters())
    scheduler   = get_linear_schedule_with_warmup(optimizer, 1000, dataset_train)
    loss        = torch_nn.MSELoss()


    best_lost_test  = 10e5
    for n_epoch in range(args.n_epochs):
        
        loss_train  = train(dataset_train, model, optimizer, loss, scheduler)

        # loss_test   = eval(dataset_test, model, loss)

        # if loss_test < best_lost_test:
        #     best_lost_test = loss_test
        #     self.save_model(model)

        print(f"epoch {n_epoch:2d} | train: loss: {loss_train:7.3f}")

        break
