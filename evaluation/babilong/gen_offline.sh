export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

data_name=$1
task_name=$2
model_path=$3
benchmark_dir=$4
out_dir=$5
benchmark_name=$6
num_gpus=$7
dialogue_turn=$8

mkdir -p $out_dir

python inference.py \
    --model_path $model_path \
    --data_dir $benchmark_dir \
    --save_path ${out_dir}/${data_name}/${task_name} \
    --task_name $task_name \
    --dataset_name $data_name \
    --benchmark_name $benchmark_name \
    --max_workers 24 \
    --max_model_len 130000 \
    --tp_size 2 \
    --dialogue_turn $dialogue_turn \
    --num_gpus $num_gpus;