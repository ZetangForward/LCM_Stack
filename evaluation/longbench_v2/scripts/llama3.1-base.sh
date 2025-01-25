python main.py \
    --model_path="Crystalcareai/meta-llama-3.1-8b" \
    --adapter_path='' \
    --tp_size=2 \
    --save_path="./results/Llama3.1-sft-longbench";

# echo "generate finish ..., begin to evaluate ..."

# python eval_longbench.py --pred_path=${SAVE_PATH}/${MODEL_NAME}
