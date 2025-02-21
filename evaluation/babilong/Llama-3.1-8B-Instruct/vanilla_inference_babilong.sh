export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1


python babilong_vllm_inference.py \
    --model_path "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --data_dir "/mnt/petrelfs/tangzecheng/local_data/BabiLong_FactsPermutation_Benchmark" \
    --save_path "/mnt/petrelfs/tangzecheng/MyRLHF/evaluation/babilong/llama-3_1-8B-Instruct/preliminary" \
    --save_name "llama-3_1-8B-Instruct_top_n" \
    --tp_size 1 \
    --num_gpus 8 \
    --max_model_len 96000 \
    --max_workers 24 \
    --sampling_type "top_n";

