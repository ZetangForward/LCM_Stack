MODEL_PATH=$1
GPU_LIST=$2
ADAPTER_PATH=$3
TAG=$4
if [ -z "$ADAPTER_PATH" ]; then
    ADAPTER_PATH=""
fi

if [ -z "$TAG" ]; then
    TAG=""
fi

python main.py \
    --model_path=$MODEL_PATH \
    --tag=$TAG \
    --adapter_path=$ADAPTER_PATH \
    --save_path='./results' \
    --gpu_lst=$GPU_LIST &

# nohup bash scripts/run.sh "/mnt/petrelfs/tangzecheng/remote_bucket/zecheng/ckpt/pg19/Llama-3.1-8B/perturb_loss_3" "0,1,2,3" > 1.log  2>&1 &

# nohup bash scripts/run.sh "/mnt/petrelfs/tangzecheng/remote_bucket/zecheng/ckpt/pg19/Llama-3.1-8B/perturb_loss" "4,5,6,7" > 2.log  2>&1 &

# nohup bash scripts/run.sh "/mnt/petrelfs/tangzecheng/remote_bucket/zecheng/ckpt/pg19/Llama-3.1-8B/real_opposite_gradient_large_pos-1e-3-v2" "0,1,2,3" > 3.log  2>&1 &

# nohup bash scripts/run.sh "/mnt/hwfile/opendatalab/tangzecheng/local_ckpt/baseline/llama3.1-8B-pg19-longce/200ep" "4,5,6,7" > 4.log  2>&1 &

# nohup bash scripts/run.sh "/mnt/hwfile/opendatalab/tangzecheng/local_ckpt/pg19/Llama-3.1-8B/opposite_gradient_large_pos-1e-3-v2/global_step300"  "0,1,2,3" > 5.log  2>&1 &

# nohup bash scripts/run.sh "/mnt/hwfile/opendatalab/tangzecheng/local_ckpt/pg19/Llama-3.1-8B/cd_lm_full-0.005/global_step300" "4,5,6,7" > 6.log  2>&1 &


# nohup bash scripts/run.sh "/mnt/petrelfs/tangzecheng/remote_bucket/zecheng/ckpt/long-context-training-V1/Llama-3.1-8B-Instruct/full"  "0,1,2,3" > full.log  2>&1 &

#重测
# nohup bash scripts/run.sh "meta-llama/Meta-Llama-3.1-8B-Instruct" "4,5,6,7" "/mnt/petrelfs/tangzecheng/remote_bucket/zecheng/ckpt/long-context-training-V1/Llama-3.1-8B-Instruct/lora" > lora.log  2>&1 &




# ------ new 

#重测
# nohup bash scripts/run.sh "meta-llama/Meta-Llama-3-8B" "0,1,2,3" "/mnt/petrelfs/tangzecheng/local_ckpt/pg19/Llama-3-8B-Scaling-CE/lora/global_step100_hf" > scale.log  2>&1 &



# nohup bash scripts/run.sh "/mnt/petrelfs/tangzecheng/local_ckpt/Llama-3-8B-ProLong-64k-Instruct" "1" > prolong.log  2>&1 &



# nohup bash scripts/run.sh "/mnt/hwfile/opendatalab/tangzecheng/local_ckpt/baseline/llama3.1-8B-pg19-ce/global_step150_hf"  "2" > step150.log  2>&1 &

# nohup bash scripts/run.sh "/mnt/hwfile/opendatalab/tangzecheng/local_ckpt/baseline/llama3.1-8B-pg19-longce/200ep"  "3" > 20ep.log  2>&1 &


# nohup bash scripts/run.sh "/mnt/petrelfs/tangzecheng/local_ckpt/Llama-3-8B-ProLong-64k-Base"  "4,5" > pro_base.log  2>&1 &

#  remains ------------------
# nohup bash scripts/run.sh "/mnt/hwfile/opendatalab/tangzecheng/local_ckpt/pg19/Llama-3.1-8B/cd_lm_full-0.005/global_step300"  "6" > step300.log  2>&1 &


# nohup bash scripts/run.sh "/mnt/hwfile/opendatalab/tangzecheng/local_ckpt/pg19/Llama-3.1-8B/cd_lm_full-0.01/global_step250"  "7" > full_step250.log  2>&1 &


# remains: ---------------


# nohup bash scripts/run.sh "meta-llama/Meta-Llama-3-8B-Instruct" "0,1,2,3" "/mnt/petrelfs/tangzecheng/remote_bucket/zecheng/ckpt/pg19/Llama-3-8B-Scaling-Noise/lora/global_step200" > noise_lora.log  2>&1 &


#  nohup bash scripts/run.sh "Crystalcareai/meta-llama-3.1-8b" "4,5,6,7" > base3_1.log  2>&1 &


# nohup bash scripts/run.sh "meta-llama/Meta-Llama-3-8B-Instruct" "4,5,6,7" > base_3.log  2>&1 &


# nohup bash scripts/run.sh "/data/zecheng/ckpt/long-context-training-V2/Qwen2.5-7B-Instruct/full_v1/global_step300" "4,5,6,7" > qwen_CD.log  2>&1 &