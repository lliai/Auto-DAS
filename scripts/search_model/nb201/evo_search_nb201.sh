#!/bin/bash
# Conduct series experiments for different search space structure

CURRENT_TIME=`date "+%Y_%m_%d"`
DATA_PATH='./data'
SAVE_DIR=./output/search_model/nb201
ITERATIONS=1000
POPU_SIZE=50
SEEDs="555"
FLOPS=50

mkdir -p ${SAVE_DIR}
echo "Current time: $CURRENT_TIME"

for SEED in ${SEEDs}
    do
        # modify DEVICE ID here for different GPU
        CUDA_VISIBLE_DEVICES=2 python exps/search_model/nb201/evo_search_model.py --iterations ${ITERATIONS} --flops ${FLOPS} 2>&1 | tee -a ${SAVE_DIR}/nb201_evo_search_jointly_${SEED}_${CURRENT_TIME}_${POPU_SIZE}_${SCALE}_${FLOPS}.log
    done

# cost time
echo "Cost time: $SECONDS seconds"
