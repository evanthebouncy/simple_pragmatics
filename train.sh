#!/bin/bash
# sample training script

# folder will be created by script. if it already exists, there will be an error
folder="./debug" 
python train_blackbox.py \
    --seed 0 --save $folder --ckpt $folder/model.pt \
    --nhid 25 --lr 0.01 --n_iter 2000