python main.py \
    --model_path="/mnt/hwfile/opendatalab/tangzecheng/local_ckpt/merge_v1_w_clues/Llama-3.1-8B/ins_adjust_weight_base_full-0.05/global_step250" \
    --adapter_path="" \
    --gpu_lst="0,1,2,3,4,5,6,7" \
    --tag="ins_adjust_weight_base_full-0.05-250steps" \
    --model_max_length_setting="normal_setting" \
    --chat_template="User: {}\nAssistant: " \
    --save_path="./results/Llama3.1-8B";