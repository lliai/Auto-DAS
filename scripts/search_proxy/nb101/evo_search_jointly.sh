#!/bin/bash
# Conduct series experiments for different search space structure

CURRENT_TIME=`date "+%Y_%m_%d"`
DATA_PATH='./data'
SAVE_DIR=./output/evo_search_jointly_nb101
ITERATIONS=5
POPU_SIZE=5
SEEDs="555"

mkdir -p ${SAVE_DIR}
echo "Current time: $CURRENT_TIME"


SCALE=True
for SEED in ${SEEDs}
    do
        # modify DEVICE ID here for different GPU
        CUDA_VISIBLE_DEVICES=0 python exps/search/nb101/evo_search_jointly.py --iterations ${ITERATIONS} --seed ${SEED} --data_path ${DATA_PATH} --popu_size ${POPU_SIZE} --scale ${SCALE} 2>&1 | tee -a ${SAVE_DIR}/nb101_evo_search_jointly_${SEED}_${CURRENT_TIME}_${POPU_SIZE}_${SCALE}.log
    done

# SCALE=False
# for SEED in ${SEEDs}
#     do
#         # modify DEVICE ID here for different GPU
#         CUDA_VISIBLE_DEVICES=0 python exps/search/nb101/evo_search_jointly.py --iterations ${ITERATIONS} --seed ${SEED} --data_path ${DATA_PATH} --popu_size ${POPU_SIZE} --scale ${SCALE} 2>&1 | tee -a ${SAVE_DIR}/nb101_evo_search_jointly_${SEED}_${CURRENT_TIME}_${POPU_SIZE}_${SCALE}.log
#     done

# cost time
echo "Cost time: $SECONDS seconds"
