#!/bin/bash

current_time=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_DIR=logs/admm_syn/
mkdir -p $LOG_DIR
task_name=SynSST2 # SynSST2, SynTwitterEmotion, SynRottenTomatoes, SynIMDB, SynRTPolarity
MODEL=microsoft/phi-1_5
# Insert synthetic data paths here
list_syn_data_path=(
    
)
num_train=100
max_steps=200
per_device_train_batch_size=16
gradient_accumulation_steps=1
LIST_TRAIN_SET_SEED=(
    0
)
kept_eval_as_train=False
num_eval_to_keep=100

LIST_LR=(
    0.000007
    0.00001
    0.000015
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

# Function to run training job
run_training_job() {
    local syn_data_path="$1"
    local lr="$2"
    local gpu="$3"
    local train_set_seed="$4"

    # Tag generation
    local synthetic_tag
    synthetic_tag=$(echo -n "$syn_data_path" | md5sum | awk '{print $1}')
    echo "Training Tag: $synthetic_tag"

    local training_tag="kept_eval_as_train=${kept_eval_as_train}_num_eval_to_keep=${num_eval_to_keep}_lr${lr}_seed${train_set_seed}"
    echo "Training Tag: $training_tag"

    local full_tag="${synthetic_tag}_${training_tag}"
    echo "Full Tag: $full_tag"
    local output_dir=./synthetic_data_FT/${current_time}/result/${full_tag}/output
    echo "Output Dir: $output_dir"

    CUDA_VISIBLE_DEVICES=$gpu python run.py \
        --trainer regular \
        --num_dev 0 \
        --num_eval 1000 \
        --logging_steps 10 \
        --output_dir ${output_dir} \
        --tag $full_tag \
        --lr_scheduler_type linear \
        --load_best_model_at_end \
        --eval_strategy steps \
        --save_strategy steps \
        --eval_steps 50 \
        --save_steps 50 \
        --overwrite_output_dir \
        --no_save_weights \
        --save_only_model \
        --train_as_classification \
        --model_name $MODEL \
        --task_name $task_name \
        --train_set_seed $train_set_seed \
        --per_device_train_batch_size $per_device_train_batch_size \
        --gradient_accumulation_steps $gradient_accumulation_steps \
        --max_steps $max_steps \
        --learning_rate $lr \
        --kept_eval_as_train $kept_eval_as_train \
        --num_eval_to_keep $num_eval_to_keep \
        --num_train $num_train \
        --syn_data_path $syn_data_path 2>&1 | tee -a ${LOG_DIR}/${full_tag}_${current_time}.log
    
    sleep 10
}

# Parallel job execution with GPU scheduling
run_parallel_jobs() {
    local max_parallel_jobs=${#LIST_GPU[@]}
    local pids=()
    local job_count=0

    for syn_data_path in "${list_syn_data_path[@]}"; do
        for lr in "${LIST_LR[@]}"; do
            for train_set_seed in "${LIST_TRAIN_SET_SEED[@]}"; do
                # Choose GPU in a round-robin fashion
                local gpu="${LIST_GPU[$((job_count % max_parallel_jobs))]}"
                
                # Run training job in background
                run_training_job "$syn_data_path" "$lr" "$gpu" "$train_set_seed" &
                pids+=($!)

                # Increment job count
                ((job_count++))

                # Wait if we've reached max parallel jobs
                sleep 10
                if [[ $job_count -ge $max_parallel_jobs ]]; then
                    wait -n  # Wait for any background job to complete
                    
                    # Remove the completed job's PID
                    for i in "${!pids[@]}"; do
                        if ! kill -0 "${pids[i]}" 2>/dev/null; then
                            unset 'pids[i]'
                        fi
                    done
                fi
            done
        done
    done

    # Wait for remaining jobs to complete
    wait
}

# Calculate total number of combinations
total_combinations=$((${#list_syn_data_path[@]} * ${#LIST_LR[@]} * ${#LIST_TRAIN_SET_SEED[@]}))

# Execute parallel jobs
run_parallel_jobs

echo "Total Combinations: $total_combinations"
echo "All training jobs completed."