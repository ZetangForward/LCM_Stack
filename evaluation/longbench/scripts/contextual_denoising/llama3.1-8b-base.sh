python main.py \
    --model_path="/mnt/hwfile/opendatalab/tangzecheng/local_ckpt/pg19/Llama-3.1-8B/cd_lm_full-0.005/global_step300" \
    --adapter_path="" \
    --gpu_lst="0,1,2,3,4,5,6,7" \
    --tag="cd_lm_full-0.005" \
    --model_max_length_setting="normal_setting" \
    --save_path="./results/Llama3.1-8B";

# echo "generate finish ..., begin to evaluate ..."

# python eval_longbench.py --pred_path=${SAVE_PATH}/${MODEL_NAME}