python ../main.py \
    --model_path="/data/hf_models/Meta-Llama-3.1-8B-Instruct" \
    --adapter_path='' \
    --model_max_length_setting="ultra_long_setting" \
    --save_path="./results/Meta-Llama-3.1-8B-Instruct";

# echo "generate finish ..., begin to evaluate ..."

# python eval_longbench.py --pred_path=${SAVE_PATH}/${MODEL_NAME}
