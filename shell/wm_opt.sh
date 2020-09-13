#!/usr/bin/env bash

model_path='ocr_model/4.ckpt.json'
batch_size=100
clip_min=0.0
clip_max=1.0
max_iter=1000
binary_search_step=1
init_const=10
wm0_path="Arial-easy-l2-eps0.2-ieps5.0-iter1000.pkl"

CUDA_N=1
font_name=('Courier' 'Georgia' 'Helvetica' 'Times' 'Arial')
case=('easy' 'random' 'hard' 'insert' 'delete' 'replace-full-word')

CUDA_VISIBLE_DEVICES=$CUDA_N python wm_opt.py --model_path=${model_path} \
                                              --font_name=${font_name[4]} \
                                              --case=${case[0]} \
                                              --batch_size=${batch_size} \
                                              --clip_min=${clip_min} \
                                              --clip_max=${clip_max} \
                                              --max_iter=${max_iter} \
                                              --binary_search_step=${binary_search_step} \
                                              --init_const=${init_const} \
                                              --wm0_path=${wm0_path}