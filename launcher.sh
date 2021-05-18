#!/bin/bash

mkdir -p backup

# echo "1. Decompose each document in dataset into paragraphs."
# python -m modules.data_reading.data_reading\
#     --num_proc 8\
#     --is_debug True

# echo "2. Start training"
CUDA_VISIBLE_DEVICES=0 python -m modules.narrativepipeline.NarrativePipeline\
    --batch 10 \
    --num_proc 8 \
    --n_epochs 60 \
    --lr 5e-4 \
    --w_decay 0.95
    # --is_debug True
