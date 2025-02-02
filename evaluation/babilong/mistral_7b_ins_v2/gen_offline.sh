export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

data_name=$1
task_name=$2
model_path=$3
benchmark_dir=$4
out_dir=$5
benchmark_name=$6

mkdir -p $out_dir

python inference.py \
    --model_path $model_path \
    --data_dir $benchmark_dir \
    --save_path ${out_dir}/${data_name}/${task_name} \
    --task_name $task_name \
    --dataset_name $data_name \
    --benchmark_name $benchmark_name \
    --max_workers 24 \
    --max_model_len 128000 \
    --tp_size 2 \
    --num_gpus 8;

# bash gen_offline.sh rapid_multiple_api tool_location meta-llama/Meta-Llama-3.1-8B-Instruct /mnt/hwfile/opendatalab/tangzecheng/benchmark_data /mnt/petrelfs/tangzecheng/local_data/inference_results/llama-3_1-8B-Instruct api

# bash gen_offline.sh rapid_parallel_api tool_location meta-llama/Meta-Llama-3.1-8B-Instruct /mnt/hwfile/opendatalab/tangzecheng/benchmark_data /mnt/petrelfs/tangzecheng/local_data/inference_results/llama-3_1-8B-Instruct api

# bash gen_offline.sh rapid_single_api tool_location meta-llama/Meta-Llama-3.1-8B-Instruct /mnt/hwfile/opendatalab/tangzecheng/benchmark_data /mnt/petrelfs/tangzecheng/local_data/inference_results/llama-3_1-8B-Instruct api

# bash gen_offline.sh babilong reasoning meta-llama/Meta-Llama-3.1-8B-Instruct "RMT-team/babilong" /mnt/petrelfs/tangzecheng/local_data/inference_results/llama-3_1-8B-Instruct babilong

# bash gen_offline.sh rapid_multiple_api tool_calling
# bash gen_offline.sh rapid_parallel_api tool_calling
# bash gen_offline.sh rapid_single_api tool_calling
# bash gen_offline.sh rapid_parallel_api tool_location
# bash gen_offline.sh rapid_multiple_api tool_location
# bash gen_offline.sh rapid_single_api tool_location