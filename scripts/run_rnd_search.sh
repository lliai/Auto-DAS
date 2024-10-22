#!/bin/bash

# Run random search for 1000 iterations

CUDA_VISIBLE_DEVICES=0 python ./diswotv2/searcher/rnd_kd_searcher.py --iterations 1000 --learning_rate 0.001
