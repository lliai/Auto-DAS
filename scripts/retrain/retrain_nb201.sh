#!/bin/bash

SAVE_DIR=./output/retrain/nb201

mkdir -p ${SAVE_DIR}

# for cifar100

CUDA_VISIBLE_DEVICES=2 python ./exps/kd_bench/train_nb201_student.py --path_t ./save/resnet110_vanilla/ckpt_epoch_240.pth --distill kd --dataset cifar100 -r 0.1 -a 0.9 -b 0  --trial 1  --batch_size 256 --learning_rate 0.1 --epochs 90 --arch_index 5292 > ${SAVE_DIR}/arch_idx_5292_run1.log


CUDA_VISIBLE_DEVICES=2 python ./exps/kd_bench/train_nb201_student.py --path_t ./save/resnet110_vanilla/ckpt_epoch_240.pth --distill kd --dataset cifar100 -r 0.1 -a 0.9 -b 0  --trial 1  --batch_size 256 --learning_rate 0.1 --epochs 90 --arch_index 5292 > ${SAVE_DIR}/arch_idx_5292_run2.log


CUDA_VISIBLE_DEVICES=2 python ./exps/kd_bench/train_nb201_student.py --path_t ./save/resnet110_vanilla/ckpt_epoch_240.pth --distill kd --dataset cifar100 -r 0.1 -a 0.9 -b 0  --trial 1  --batch_size 256 --learning_rate 0.1 --epochs 90 --arch_index 5292 > ${SAVE_DIR}/arch_idx_5292_run3.log
