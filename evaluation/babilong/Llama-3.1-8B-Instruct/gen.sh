python inference_babilong.py \
    --dataset_name "RMT-team/babilong" \
    --model_path 'meta-llama/Meta-Llama-3.1-8B-Instruct' \
    --tag 'longalpaca_global_step250' \
    --adapter_path '/mnt/petrelfs/tangzecheng/remote_bucket/zecheng/ckpt/baseline/Llama-3.1-8B-Instruct/longalpaca/adapter/global_step250' \
    --save_path '/mnt/petrelfs/tangzecheng/MyRLHF/inference/babilong/gan/Llama-3.1-8B-Instruct/' \
    --gpu_id 0,1,2,3,4,5,6,7 \
    --tp_size 1;