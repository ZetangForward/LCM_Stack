python main.py \
    --model_path="/mnt/hwfile/opendatalab/tangzecheng/local_ckpt/merge_v1_w_clues/Llama-3.1-8B/ins_adjust_weight_base_full-0.05/global_step250" \
    --adapter_path='' \
    --tp_size=2 \
    --tag='ins_adjust_weight_base_full-0.05-global_step250' \
    --save_path="./results/contextual_denoising";

# echo "generate finish ..., begin to evaluate ..."

# python eval_longbench.py --pred_path=${SAVE_PATH}/${MODEL_NAME}
