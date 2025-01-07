export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

python main.py \
    --model_path="meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --adapter_path='/mnt/petrelfs/tangzecheng/local_ckpt/merge_v1_fix/Llama-3.1-8B-Instruct/sft/global_step100' \
    --model_max_length_setting="normal_setting" \
    --save_path="./results/Llama3.1";

# echo "generate finish ..., begin to evaluate ..."

# python eval_longbench.py --pred_path=${SAVE_PATH}/${MODEL_NAME}
 