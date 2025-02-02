export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

data_name=$1
task_name=$2

model_path='meta-llama/Meta-Llama-3.1-8B-Instruct'
benchmark_dir='/mnt/petrelfs/tangzecheng/local_data/benchmark_data'
out_dir='/mnt/petrelfs/tangzecheng/local_data/inference_results/llama-3_1-8B-Instruct/'${data_name}/${task_name}


mkdir -p $out_dir

python generate.py \
    --ports 4100 4101 4102 4103 4104 4105 4106 4107 \
    --model_path $model_path \
    --data_dir $benchmark_dir \
    --save_path $out_dir \
    --task_name $task_name \
    --dataset_name $data_name \
    --max_workers 16;

# bash gen.sh rapid_parallel_api tool_calling
# bash gen.sh rapid_multiple_api tool_calling
# bash gen.sh rapid_single_api tool_calling
# bash gen.sh rapid_parallel_api tool_location
# bash gen.sh rapid_multiple_api tool_location
# bash gen.sh rapid_single_api tool_location