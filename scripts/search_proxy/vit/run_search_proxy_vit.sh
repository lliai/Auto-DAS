#!/bin/bash
# Conduct series experiments for different search space structure

CURRENT_TIME=`date "+%Y_%m_%d"`
DATA_PATH='./data'
SAVE_DIR=./output/rnd_search_jointly_vit
ITERATIONS=500
SEEDs=666

mkdir -p ${SAVE_DIR}
echo "Current time: $CURRENT_TIME"

# c100 autoformer search space
# echo "autoformer search space w scale"
# CUDA_VISIBLE_DEVICES=2 python exps/search_proxy/vit/rnd_search_jointly.py --iterations ${ITERATIONS} --dataset c100 --scale True --api_path './data/diswotv2_autoformer.pth' --data_path ${DATA_PATH} 2>&1 | tee -a ${SAVE_DIR}/vit_rnd_search_jointly_cifar100_${SEED}_${CURRENT_TIME}_w_scale.log

# echo "autoformer search space wo scale"
# CUDA_VISIBLE_DEVICES=2 python exps/search_proxy/vit/rnd_search_jointly.py --iterations ${ITERATIONS} --dataset c100 --scale False --api_path './data/diswotv2_autoformer.pth' --data_path ${DATA_PATH} 2>&1 | tee -a ${SAVE_DIR}/vit_rnd_search_jointly_cifar100_${SEED}_${CURRENT_TIME}_wo_scale.log


# c100 pit search space
echo "pit search space w scale"
CUDA_VISIBLE_DEVICES=1 python exps/search_proxy/vit/rnd_search_jointly.py --iterations ${ITERATIONS} --dataset c100 --scale True --searchspace pit --api_path './data/diswotv2_pit.pth' --data_path ${DATA_PATH} 2>&1 | tee -a ${SAVE_DIR}/vit_rnd_search_jointly_cifar100_${SEED}_${CURRENT_TIME}_w_scale.log

echo "pit search space wo scale"
CUDA_VISIBLE_DEVICES=1 python exps/search_proxy/vit/rnd_search_jointly.py --iterations ${ITERATIONS} --dataset c100 --scale True --searchspace pit --api_path './data/diswotv2_pit.pth' --data_path ${DATA_PATH} 2>&1 | tee -a ${SAVE_DIR}/vit_rnd_search_jointly_cifar100_${SEED}_${CURRENT_TIME}_wo_scale.log

echo "Done time: $CURRENT_TIME"
