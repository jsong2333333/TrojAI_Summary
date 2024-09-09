#!/bin/bash
for model_num in  0 4 6 7 12 15 16 19 24 27 28 29 33 34 35 36 42 43 44 47 51 53 54 62 66 69 71 72 74 79 81 82 83 84 90 92 93 101 102 110 111 112 114
do
    CUDA_VISIBLE_DEVICES=3 python main_cka_debug.py --model_num $model_num --clean_poisoned_input global_poisoned --layer_types fulllayer
done