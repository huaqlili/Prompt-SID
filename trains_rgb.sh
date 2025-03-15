#!/usr/bin/env bash


CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=4392 Diffneighbors/train.py -opt options/train_imagenet.yml --launcher pytorch