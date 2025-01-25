python main.py \
    --model_path="/mnt/hwfile/opendatalab/tangzecheng/local_ckpt/pg19/Llama-3.1-8B/opposite_gradient_large_pos-1e-3-v2/global_step300" \
    --adapter_path="" \
    --gpu_lst="0,1,2,3,4,5,6,7" \
    --tag="opposite_gradient_large_pos-1e-3-v2-300steps" \
    --model_max_length_setting="normal_setting" \
    --chat_template="User: {}\nAssistant: " \
    --save_path="./results/Llama3.1-8B";