
# STEPS=(117 234 351 468 585 702 819 936 1053 1170)  # You can set multiple steps: STEPS=(234 235 236) etc.
STEPS=(125)  # You can set multiple steps: STEPS=(234 235 236) etc.

# MODEL_NAME=metamath-sft-gemma-2-2b
# MODEL_NAME=mutual_metaninstruct1-sft-qwen-2.5-0.5b
# MODEL_NAME=mutual_metaninstruct1-sft-qwen-2.5-0.5b
MODEL_NAME=metaninstruct1-sft-evolm-1b

for STEP in "${STEPS[@]}"; do
    echo $STEP
    ckpt_dir=/data/hnn5071/lrm/verl-imitation/runs/${MODEL_NAME}/global_step_${STEP}

    python3 -m verl.model_merger merge \
        --backend fsdp \
        --local_dir "$ckpt_dir" \
        --target_dir "${ckpt_dir}/huggingface"

    # backup_dir=/home/nlp/hnn5071/models/${MODEL_NAME}/global_step_${STEP}
    # mkdir -p "$backup_dir"
    # cp -r "${ckpt_dir}/huggingface" "$backup_dir"
done
