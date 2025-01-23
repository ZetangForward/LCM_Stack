python main.py \
    --model_path="/mnt/hwfile/opendatalab/tangzecheng/local_ckpt/baseline/llama3.1-8B-sft/global_step150" \
    --adapter_path="" \
    --gpu_lst="0,1,2,3,4,5,6,7" \
    --tag="baseline_sft" \
    --chat_template="User: {}\nAssistant: " \
    --model_max_length_setting="normal_setting" \
    --save_path="./results/Llama3.1-8B";

# echo "generate finish ..., begin to evaluate ..."

# python eval_longbench.py --pred_path=${SAVE_PATH}/${MODEL_NAME}