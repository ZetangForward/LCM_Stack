python inference_babilong.py \
    --dataset_name "RMT-team/babilong" \
    --model_path '/data/hf_models/Meta-Llama-3.1-8B-Instruct' \
    --tag gan_nce_global_step675_longmit \
    --adapter_path '' \
    --save_path '/mnt/petrelfs/tangzecheng/MyRLHF/inference/babilong/gan/Llama-3.1-8B-Instruct/' \
    --gpu_id 0,1,2,3,4,5,6,7 \
    --tp_size 1;
