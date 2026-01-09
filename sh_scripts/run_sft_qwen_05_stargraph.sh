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
    data.train_files=/data/hnn5071/lrm/verl-imitation/data/star_graph/train.parquet \
    data.val_files=/data/hnn5071/lrm/verl-imitation/data/star_graph/test.parquet \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    data.max_length=1024 \
    data.truncation=right \
    data.train_batch_size=256 \
    optim.lr=1e-5 \
    data.prompt_dict_keys=['question'] \
    +data.response_dict_keys=['answer'] \
    data.micro_batch_size_per_gpu=8 \
    model.partial_pretrain=Qwen/Qwen2.5-0.5B \
    trainer.default_local_dir=$save_path \
    trainer.project_name=star_graph-sft \
    trainer.experiment_name=star_graph-sft-qwen-2.5-0.5b \
    trainer.logger=['console','wandb'] \
    trainer.total_epochs=10 $@ \
    trainer.save_freq=100 \
    use_remove_padding=true
