python main.py \
    --model_path="Crystalcareai/meta-llama-3.1-8b" \
    --adapter_path="" \
    --gpu_lst="0,1,3,4" \
    --tag="" \
    --model_max_length_setting="normal_setting" \
    --save_path="./results/Llama3.1-8B";

# echo "generate finish ..., begin to evaluate ..."

# python eval_longbench.py --pred_path=${SAVE_PATH}/${MODEL_NAME}