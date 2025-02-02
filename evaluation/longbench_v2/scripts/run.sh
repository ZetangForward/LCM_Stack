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

# nohup bash scripts/run.sh "/mnt/hwfile/opendatalab/tangzecheng/local_ckpt/baseline/llama3.1-8B-pg19-longce/200ep" "4,5,6,7sbsBs" > 4.log  2>&1 &

# nohup bash scripts/run.sh "/mnt/hwfile/opendatalab/tangzecheng/local_ckpt/pg19/Llama-3.1-8B/opposite_gradient_large_pos-1e-3-v2/global_step300"  "0,1,2,3" > 5.log  2>&1 &

# nohup bash scripts/run.sh "/mnt/hwfile/opendatalab/tangzecheng/local_ckpt/pg19/Llama-3.1-8B/cd_lm_full-0.005/global_step300" "4,5,6,7" > 6.log  2>&1 &
