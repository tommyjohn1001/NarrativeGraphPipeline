#!/bin/bash

mkdir -p backup

# echo "1. Decompose each document in dataset into paragraphs."
# python -m modules.data_reading.data_reading --num_proc 4

# echo "2. Start training"
python -m modules.narrativepipeline.NarrativePipeline\
    --batch 30 \
    --num_proc 8 \
    --n_epochs 60 \
    --lr 7e-5 \
    --w_decay 1e-4 \
