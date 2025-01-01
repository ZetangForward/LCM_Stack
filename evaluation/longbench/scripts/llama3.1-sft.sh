SAVE_PATH="./longbench"
CONTEXT_SETTING="normal_setting"
MODEL_NAME=s1-tmp2-${CONTEXT_SETTING}
MODEL_PATH="/data/zecheng/hf_models/Meta-Llama-3-8B-Instruct"
PEFT_PATH="/data/zecheng/ckpt/s1-tmp2/ckpt-1050/context_scaling"

python main.py \
    --model_path="meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --adapter_path='/mnt/petrelfs/tangzecheng/local_ckpt/merge_v1_fix/Llama-3.1-8B-Instruct/sft_ctx_loss/global_step200' \
    --model_max_length_setting="normal_setting" \
    --save_path="./results/Llama3.1-sft";

# echo "generate finish ..., begin to evaluate ..."

# python eval_longbench.py --pred_path=${SAVE_PATH}/${MODEL_NAME}
