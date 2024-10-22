#!/bin/bash

CURRENT_TIME=`date "+%Y_%m_%d"`
SAVE_DIR=./output/rank/naive_zc

mkdir -p ${SAVE_DIR}
echo "Current time: $CURRENT_TIME"

# ZC: flops, fisher, grad_norm, snip, synflow, nwot

#### NB101-KD ####


ZC=flops
SEED=111
CUDA_VISIBLE_DEVICES=0 python exps/rank/nb101/eval_rank_zc.py --zc_name ${ZC} --seed ${SEED}  > ${SAVE_DIR}/eval_NB101_rank_naive_${ZC}_${SEED}_${CURRENT_TIME}.txt 2>&1

SEED=222
CUDA_VISIBLE_DEVICES=0 python exps/rank/nb101/eval_rank_zc.py --zc_name ${ZC} --seed ${SEED}  > ${SAVE_DIR}/eval_NB101_rank_naive_${ZC}_${SEED}_${CURRENT_TIME}.txt 2>&1

SEED=333
CUDA_VISIBLE_DEVICES=0 python exps/rank/nb101/eval_rank_zc.py --zc_name ${ZC} --seed ${SEED}  > ${SAVE_DIR}/eval_NB101_rank_naive_${ZC}_${SEED}_${CURRENT_TIME}.txt 2>&1

ZC=fisher
SEED=111
CUDA_VISIBLE_DEVICES=0 python exps/rank/nb101/eval_rank_zc.py --zc_name ${ZC} --seed ${SEED}  > ${SAVE_DIR}/eval_NB101_rank_naive_${ZC}_${SEED}_${CURRENT_TIME}.txt 2>&1

SEED=222
CUDA_VISIBLE_DEVICES=0 python exps/rank/nb101/eval_rank_zc.py --zc_name ${ZC} --seed ${SEED}  > ${SAVE_DIR}/eval_NB101_rank_naive_${ZC}_${SEED}_${CURRENT_TIME}.txt 2>&1

SEED=333
CUDA_VISIBLE_DEVICES=0 python exps/rank/nb101/eval_rank_zc.py --zc_name ${ZC} --seed ${SEED}  > ${SAVE_DIR}/eval_NB101_rank_naive_${ZC}_${SEED}_${CURRENT_TIME}.txt 2>&1

ZC=grad_norm
SEED=111
CUDA_VISIBLE_DEVICES=0 python exps/rank/nb101/eval_rank_zc.py --zc_name ${ZC} --seed ${SEED}  > ${SAVE_DIR}/eval_NB101_rank_naive_${ZC}_${SEED}_${CURRENT_TIME}.txt 2>&1

SEED=222
CUDA_VISIBLE_DEVICES=0 python exps/rank/nb101/eval_rank_zc.py --zc_name ${ZC} --seed ${SEED}  > ${SAVE_DIR}/eval_NB101_rank_naive_${ZC}_${SEED}_${CURRENT_TIME}.txt 2>&1

SEED=333
CUDA_VISIBLE_DEVICES=0 python exps/rank/nb101/eval_rank_zc.py --zc_name ${ZC} --seed ${SEED}  > ${SAVE_DIR}/eval_NB101_rank_naive_${ZC}_${SEED}_${CURRENT_TIME}.txt 2>&1

ZC=snip
SEED=111
CUDA_VISIBLE_DEVICES=0 python exps/rank/nb101/eval_rank_zc.py --zc_name ${ZC} --seed ${SEED}  > ${SAVE_DIR}/eval_NB101_rank_naive_${ZC}_${SEED}_${CURRENT_TIME}.txt 2>&1

SEED=222
CUDA_VISIBLE_DEVICES=0 python exps/rank/nb101/eval_rank_zc.py --zc_name ${ZC} --seed ${SEED}  > ${SAVE_DIR}/eval_NB101_rank_naive_${ZC}_${SEED}_${CURRENT_TIME}.txt 2>&1

SEED=333
CUDA_VISIBLE_DEVICES=0 python exps/rank/nb101/eval_rank_zc.py --zc_name ${ZC} --seed ${SEED}  > ${SAVE_DIR}/eval_NB101_rank_naive_${ZC}_${SEED}_${CURRENT_TIME}.txt 2>&1

ZC=synflow
SEED=111
CUDA_VISIBLE_DEVICES=0 python exps/rank/nb101/eval_rank_zc.py --zc_name ${ZC} --seed ${SEED}  > ${SAVE_DIR}/eval_NB101_rank_naive_${ZC}_${SEED}_${CURRENT_TIME}.txt 2>&1

SEED=222
CUDA_VISIBLE_DEVICES=0 python exps/rank/nb101/eval_rank_zc.py --zc_name ${ZC} --seed ${SEED}  > ${SAVE_DIR}/eval_NB101_rank_naive_${ZC}_${SEED}_${CURRENT_TIME}.txt 2>&1

SEED=333
CUDA_VISIBLE_DEVICES=0 python exps/rank/nb101/eval_rank_zc.py --zc_name ${ZC} --seed ${SEED}  > ${SAVE_DIR}/eval_NB101_rank_naive_${ZC}_${SEED}_${CURRENT_TIME}.txt 2>&1

ZC=nwot
SEED=111
CUDA_VISIBLE_DEVICES=0 python exps/rank/nb101/eval_rank_zc.py --zc_name ${ZC} --seed ${SEED}  > ${SAVE_DIR}/eval_NB101_rank_naive_${ZC}_${SEED}_${CURRENT_TIME}.txt 2>&1

SEED=222
CUDA_VISIBLE_DEVICES=0 python exps/rank/nb101/eval_rank_zc.py --zc_name ${ZC} --seed ${SEED}  > ${SAVE_DIR}/eval_NB101_rank_naive_${ZC}_${SEED}_${CURRENT_TIME}.txt 2>&1

SEED=333
CUDA_VISIBLE_DEVICES=0 python exps/rank/nb101/eval_rank_zc.py --zc_name ${ZC} --seed ${SEED}  > ${SAVE_DIR}/eval_NB101_rank_naive_${ZC}_${SEED}_${CURRENT_TIME}.txt 2>&1


#### NB201-KD ####

ZC=flops
SEED=111
CUDA_VISIBLE_DEVICES=0 python exps/rank/nb201/eval_rank_zc.py --zc_name ${ZC} --seed ${SEED}  > ${SAVE_DIR}/eval_nb201_rank_naive_${ZC}_${SEED}_${CURRENT_TIME}.txt 2>&1

SEED=222
CUDA_VISIBLE_DEVICES=0 python exps/rank/nb201/eval_rank_zc.py --zc_name ${ZC} --seed ${SEED}  > ${SAVE_DIR}/eval_nb201_rank_naive_${ZC}_${SEED}_${CURRENT_TIME}.txt 2>&1

SEED=333
CUDA_VISIBLE_DEVICES=0 python exps/rank/nb201/eval_rank_zc.py --zc_name ${ZC} --seed ${SEED}  > ${SAVE_DIR}/eval_nb201_rank_naive_${ZC}_${SEED}_${CURRENT_TIME}.txt 2>&1

ZC=fisher
SEED=111
CUDA_VISIBLE_DEVICES=0 python exps/rank/nb201/eval_rank_zc.py --zc_name ${ZC} --seed ${SEED}  > ${SAVE_DIR}/eval_nb201_rank_naive_${ZC}_${SEED}_${CURRENT_TIME}.txt 2>&1

SEED=222
CUDA_VISIBLE_DEVICES=0 python exps/rank/nb201/eval_rank_zc.py --zc_name ${ZC} --seed ${SEED}  > ${SAVE_DIR}/eval_nb201_rank_naive_${ZC}_${SEED}_${CURRENT_TIME}.txt 2>&1

SEED=333
CUDA_VISIBLE_DEVICES=0 python exps/rank/nb201/eval_rank_zc.py --zc_name ${ZC} --seed ${SEED}  > ${SAVE_DIR}/eval_nb201_rank_naive_${ZC}_${SEED}_${CURRENT_TIME}.txt 2>&1

ZC=grad_norm
SEED=111
CUDA_VISIBLE_DEVICES=0 python exps/rank/nb201/eval_rank_zc.py --zc_name ${ZC} --seed ${SEED}  > ${SAVE_DIR}/eval_nb201_rank_naive_${ZC}_${SEED}_${CURRENT_TIME}.txt 2>&1

SEED=222
CUDA_VISIBLE_DEVICES=0 python exps/rank/nb201/eval_rank_zc.py --zc_name ${ZC} --seed ${SEED}  > ${SAVE_DIR}/eval_nb201_rank_naive_${ZC}_${SEED}_${CURRENT_TIME}.txt 2>&1

SEED=333
CUDA_VISIBLE_DEVICES=0 python exps/rank/nb201/eval_rank_zc.py --zc_name ${ZC} --seed ${SEED}  > ${SAVE_DIR}/eval_nb201_rank_naive_${ZC}_${SEED}_${CURRENT_TIME}.txt 2>&1

ZC=snip
SEED=111
CUDA_VISIBLE_DEVICES=0 python exps/rank/nb201/eval_rank_zc.py --zc_name ${ZC} --seed ${SEED}  > ${SAVE_DIR}/eval_nb201_rank_naive_${ZC}_${SEED}_${CURRENT_TIME}.txt 2>&1

SEED=222
CUDA_VISIBLE_DEVICES=0 python exps/rank/nb201/eval_rank_zc.py --zc_name ${ZC} --seed ${SEED}  > ${SAVE_DIR}/eval_nb201_rank_naive_${ZC}_${SEED}_${CURRENT_TIME}.txt 2>&1

SEED=333
CUDA_VISIBLE_DEVICES=0 python exps/rank/nb201/eval_rank_zc.py --zc_name ${ZC} --seed ${SEED}  > ${SAVE_DIR}/eval_nb201_rank_naive_${ZC}_${SEED}_${CURRENT_TIME}.txt 2>&1

ZC=synflow
SEED=111
CUDA_VISIBLE_DEVICES=0 python exps/rank/nb201/eval_rank_zc.py --zc_name ${ZC} --seed ${SEED}  > ${SAVE_DIR}/eval_nb201_rank_naive_${ZC}_${SEED}_${CURRENT_TIME}.txt 2>&1

SEED=222
CUDA_VISIBLE_DEVICES=0 python exps/rank/nb201/eval_rank_zc.py --zc_name ${ZC} --seed ${SEED}  > ${SAVE_DIR}/eval_nb201_rank_naive_${ZC}_${SEED}_${CURRENT_TIME}.txt 2>&1

SEED=333
CUDA_VISIBLE_DEVICES=0 python exps/rank/nb201/eval_rank_zc.py --zc_name ${ZC} --seed ${SEED}  > ${SAVE_DIR}/eval_nb201_rank_naive_${ZC}_${SEED}_${CURRENT_TIME}.txt 2>&1

ZC=nwot
SEED=111
CUDA_VISIBLE_DEVICES=0 python exps/rank/nb201/eval_rank_zc.py --zc_name ${ZC} --seed ${SEED}  > ${SAVE_DIR}/eval_nb201_rank_naive_${ZC}_${SEED}_${CURRENT_TIME}.txt 2>&1

SEED=222
CUDA_VISIBLE_DEVICES=0 python exps/rank/nb201/eval_rank_zc.py --zc_name ${ZC} --seed ${SEED}  > ${SAVE_DIR}/eval_nb201_rank_naive_${ZC}_${SEED}_${CURRENT_TIME}.txt 2>&1

SEED=333
CUDA_VISIBLE_DEVICES=0 python exps/rank/nb201/eval_rank_zc.py --zc_name ${ZC} --seed ${SEED}  > ${SAVE_DIR}/eval_nb201_rank_naive_${ZC}_${SEED}_${CURRENT_TIME}.txt 2>&1
