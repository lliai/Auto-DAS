#!/bin/bash

CURRENT_TIME=`date "+%Y_%m_%d"`
DATA_PATH='./data'
SAVE_DIR=./output/search_model/zen
ITERATIONS=5000
SEEDs="555"


mkdir -p ${SAVE_DIR}
echo "Current time: $CURRENT_TIME"

# Zen TE-NAS Syncflow
# res18 13M
CUDA_VISIBLE_DEVICES=0 python exps/search_model/zen/evo_search_model.py  \
    --input_image_size 224 \
    --max_layers 18 \
    --num_classes 1000 \
    --save_dir ${SAVE_DIR}/run_evo_diswot_res18_${SEEDs} \
    --budget_model_size 13e6 \
    --evolution_max_iter ${ITERATIONS} \
    --zero_shot_score DISWOTv2
    #  | tee ${SAVE_DIR}/DISWOTv2_res18_13M_${SEEDs}.log
