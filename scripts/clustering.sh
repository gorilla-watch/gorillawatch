#!/bin/bash

python constrained_clustering.py \
    --image_dir /workspaces/vast-gorilla/gorillawatch/data/eval_cleaned_10_individuals_open/val \
    --checkpoint_path /workspaces/gorillawatch/new_checkpoints/giant_clean_10_individuals_open.pth \
    --batch_size 32 \
    --constrained \
    --thresholds 0.0001 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 4 7 10 13 16 19 22 25 28 31 34 37 40 43 46 49 52 55 58 61 64 67 70 73 76 79 82 85 88 91 94 97
