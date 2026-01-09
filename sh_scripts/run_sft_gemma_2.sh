set -x

if [ "$#" -lt 2 ]; then
    echo "Usage: run_qwen_05_sp2.sh <nproc_per_node> <save_path> [other_configs...]"
    exit 1
fi

nproc_per_node=$1
save_path=$2

# Shift the arguments so $@ refers to the rest
shift 2

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=/data/hnn5071/lrm/verl-imitation/data/metamath/train.parquet \
    data.val_files=/data/hnn5071/lrm/verl-imitation/data/metamath/test.parquet \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    data.max_length=1024 \
    data.truncation=right \
    data.train_batch_size=256 \
    optim.lr=1e-5 \
    data.prompt_dict_keys=['question'] \
    +data.response_dict_keys=['answer'] \
    data.micro_batch_size_per_gpu=2 \
    model.partial_pretrain=google/gemma-2-2b \
    model.tokenizer_path=google/gemma-2-2b-it \
    trainer.default_local_dir=$save_path \
    trainer.project_name=metamath-sft \
    trainer.experiment_name=metamath-sft-gemma-2-2b \
    trainer.logger=['console','wandb'] \
    trainer.total_epochs=10 $@ \
    trainer.save_freq=117 \
    use_remove_padding=true
