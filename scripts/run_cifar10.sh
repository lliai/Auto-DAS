#!/bin/bash

# run on cifar10
# run on cifar10
CUDA_VISIBLE_DEVICES=1 CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar10 --model resnet --depth 32 --epochs 200 --schedule 60 120 160 --gammas 0.2 0.2 0.2 --wd 5e-4 --save cifar10

# Path: exps\run_cifar100.sh
#!/bin/bash

# run on cifar100
 # run on cifar100
 CUDA_VISIBLE_DEVICES=1 CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar100 --model resnet --depth 32 --epochs 200 --schedule 60 120 160 --gammas 0.2 0.2 0.2 --wd 5e-4 --save cifar100

# Path: exps\run_svhn.sh
#!/bin/bash

# sample scripts for running the distillation code
# use resnet32x4 and resnet8x4 as an example

# kd
CUDA_VISIBLE_DEVICES=0 python exps/train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill kd --model_s resnet8x4 -r 0.1 -a 0.9 -b 0 --trial 1
# FitNet
CUDA_VISIBLE_DEVICES=0 python exps/train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill hint --model_s resnet8x4 -a 0 -b 100 --trial 1
# AT
CUDA_VISIBLE_DEVICES=0 python exps/train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill attention --model_s resnet8x4 -a 0 -b 1000 --trial 1
# SP
CUDA_VISIBLE_DEVICES=0 python exps/train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill similarity --model_s resnet8x4 -a 0 -b 3000 --trial 1
# CC
CUDA_VISIBLE_DEVICES=0 python exps/train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill correlation --model_s resnet8x4 -a 0 -b 0.02 --trial 1
# VID
CUDA_VISIBLE_DEVICES=0 python exps/train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill vid --model_s resnet8x4 -a 0 -b 1 --trial 1
# RKD
CUDA_VISIBLE_DEVICES=0 python exps/train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill rkd --model_s resnet8x4 -a 0 -b 1 --trial 1
# PKT
CUDA_VISIBLE_DEVICES=0 python exps/train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill pkt --model_s resnet8x4 -a 0 -b 30000 --trial 1
# AB
CUDA_VISIBLE_DEVICES=0 python exps/train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill abound --model_s resnet8x4 -a 0 -b 1 --trial 1
# FT
CUDA_VISIBLE_DEVICES=0 python exps/train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill factor --model_s resnet8x4 -a 0 -b 200 --trial 1
# FSP
CUDA_VISIBLE_DEVICES=0 python exps/train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill fsp --model_s resnet8x4 -a 0 -b 50 --trial 1
# NST
CUDA_VISIBLE_DEVICES=0 python exps/train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill nst --model_s resnet8x4 -a 0 -b 50 --trial 1
# CRD
CUDA_VISIBLE_DEVICES=0 python exps/train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill crd --model_s resnet8x4 -a 0 -b 0.8 --trial 1

# CRD+KD
CUDA_VISIBLE_DEVICES=0 python exps/train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill crd --model_s resnet8x4 -a 1 -b 0.8 --trial 1
