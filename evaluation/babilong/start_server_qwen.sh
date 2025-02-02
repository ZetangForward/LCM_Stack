#!/bin/bash
MODEL_NAME=$1

for i in 0 1 2 3 4 5 6 7;do
    CUDA_VISIBLE_DEVICES=$i python -m vllm.entrypoints.openai.api_server \
        --model=${MODEL_NAME} \
        --served-model-name=qwen \
        --gpu-memory-utilization=0.98 \
        --max-model-len=128000 \
        --dtype=bfloat16 \
        --tensor-parallel-size 4 \
        --host 127.0.0.1 \
        --port $((4100+i)) \
        --trust-remote-code \
        --swap-space 0 &
done


# bash ./start_server.sh 
# bash start_server.sh meta-llama/Meta-Llama-3.1-8B-Instruct
# pkill -f vllm.entrypoints.openai.api_server