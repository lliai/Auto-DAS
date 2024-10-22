#!/bin/bash
# Conduct series experiments for different search space structure

CURRENT_TIME=`date "+%Y_%m_%d"`
DATA_PATH='./data'
SAVE_DIR=./output/evo_search_interaction_s1
ITERATIONS=1000
POPU_SIZE=50
SEEDs="555"

mkdir -p ${SAVE_DIR}
echo "Current time: $CURRENT_TIME"

for SEED in ${SEEDs}
    do
        # modify DEVICE ID here for different GPU
        CUDA_VISIBLE_DEVICES=2 python exps/search/s0/evo_search_interaction.py --iterations ${ITERATIONS} --seed ${SEED} --data_path ${DATA_PATH} --popu_size ${POPU_SIZE} 2>&1 | tee -a ${SAVE_DIR}/s1_evo_search_interaction_${SEED}_${CURRENT_TIME}_${POPU_SIZE}.log
    done

# cost time
echo "Cost time: $SECONDS seconds"
