python inference_babilong.py \
    --dataset_name "RMT-team/babilong" \
    --model_path 'Qwen/Qwen2.5-7B-Instruct' \
    --adapter_path '/mnt/petrelfs/tangzecheng/local_ckpt/merge_v1_fix/Qwen2.5-7B-Instruct/simpo/global_step150' \
    --save_path '/mnt/petrelfs/tangzecheng/MyRLHF/inference/babilong/merge_v1/Qwen2.5-7B-Instruct/' \
    --num_gpus 8 \
    --tp_size 1 \





# bash gen.sh rapid_parallel_api tool_calling
# bash gen.sh rapid_multiple_api tool_calling
# bash gen.sh rapid_single_api tool_calling
# bash gen.sh rapid_parallel_api tool_location
# bash gen.sh rapid_multiple_api tool_location
# bash gen.sh rapid_single_api tool_location