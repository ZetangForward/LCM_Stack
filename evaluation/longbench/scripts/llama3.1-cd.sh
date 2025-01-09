python main.py \
    --model_path="meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --adapter_path="/mnt/petrelfs/tangzecheng/remote_bucket/zecheng/ckpt/merge_v1_fix/Llama-3.1-8B-Instruct/context_denoise/global_step350" \
    --gpu_lst="1,2,3,6" \
    --model_max_length_setting="normal_setting" \
    --save_path="./results/Llama3.1";

# echo "generate finish ..., begin to evaluate ..."

# python eval_longbench.py --pred_path=${SAVE_PATH}/${MODEL_NAME}
