python main.py \
    --model_path="/mnt/hwfile/opendatalab/tangzecheng/local_ckpt/pg19/Llama-3.1-8B/opposite_gradient_large_pos-1e-3/global_step250" \
    --adapter_path="" \
    --gpu_lst="0,1,2,3,4,5,6,7" \
    --tag="opposite_gradient_large_pos-1e-3" \
    --model_max_length_setting="normal_setting" \
    --save_path="./results/Llama3.1-8B";

# echo "generate finish ..., begin to evaluate ..."

# python eval_longbench.py --pred_path=${SAVE_PATH}/${MODEL_NAME}