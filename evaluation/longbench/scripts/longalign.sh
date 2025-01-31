python main.py \
    --model_path="THUDM/LongAlign-7B-64k" \
    --model_max_length_setting="normal_setting" \
    --gpu_lst="0,1,2,3,4,5,6,7" \
    --save_path="./results/LongAlign-7B-64k" \
    --chat_template="[INST]Hi![/INST]Hello! What can I assist you today?\n[INST]{}[/INST]";

# echo "generate finish ..., begin to evaluate ..."

# python eval_longbench.py --pred_path=${SAVE_PATH}/${MODEL_NAME}
