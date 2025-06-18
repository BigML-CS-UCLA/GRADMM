#!/bin/bash

current_time=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_DIR=logs/admm/
mkdir -p $LOG_DIR
rng_seed=42
# Data and model
MODEL=phi
dataset=sst2 # sst2, rotten_tomatoes, TwitterEmotion, imdb, rtpolarity
split=validation # train, validation
n_gen_samples=100
subset_size=50 # number of real samples to use
n_gen=10
gen_bs=10
# ADMM
use_auto_gen_tokens=true
n_steps=30
topk=200
grad_clip=1.0
opt_alg=admm
admm_inner_steps=50
# DP
dp_epsilon=0.05
dp_delta=1e-4
dp_c=1.0
# Logging and saving
print_full=true
print_every=10
save_every=1
base_dir=./synthetic_data

LIST_RHO=(
    0.01
    0.1
    0.5
    1
    5
)
LIST_GPU=(
    0
    1
    2
    3
    4
    5
    6
    7
)

# Loop through grad_clip and GPUs
for i in "${!LIST_RHO[@]}"; do
    (admm_rho="${LIST_RHO[i]}"
    gpu="${LIST_GPU[i]}"
    full_tag=${dataset}-${split}-${MODEL}-nreal${subset_size}-steps${n_steps}-nsyn${n_gen}-${opt_alg}-dp_eps${dp_epsilon}-rho${admm_rho}-inner${admm_inner_steps}-seed${rng_seed}
    work_base_dir=${base_dir}/${full_tag}_${current_time}
    echo "Work Base Dir: $work_base_dir"

    CUDA_VISIBLE_DEVICES=$gpu python generate.py \
        --rng_seed $rng_seed \
        --dataset $dataset \
        --split $split \
        --batch_size $subset_size \
        --n_steps $n_steps \
        --n_gen_samples $n_gen_samples \
        --subset_size $subset_size \
        --n_gen $n_gen \
        --gen_bs $gen_bs \
        --use_auto_gen_tokens $use_auto_gen_tokens \
        --print_full $print_full \
        --print_every $print_every \
        --save_every $save_every \
        --model_name $MODEL \
        --opt_alg $opt_alg \
        --admm_rho $admm_rho \
        --admm_inner_steps $admm_inner_steps \
        --work_base_dir $work_base_dir \
        --grad_clip $grad_clip \
        --use_dp "true" \
        --dp_epsilon $dp_epsilon \
        --dp_delta $dp_delta \
        --dp_c $dp_c \
        --topk $topk 2>&1 | tee -a ${LOG_DIR}/${full_tag}_${current_time}.log) &
done