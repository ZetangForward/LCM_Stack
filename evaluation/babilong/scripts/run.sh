MODEL_PATH=$1
ADAPTER_PATH=$2
TAG=$3
GPU_ID=$4

if [ -z "$ADAPTER_PATH" ]; then
    ADAPTER_PATH=""
fi

if [ -z "$TAG" ]; then
    TAG=""
fi

if [ -z "$GPU_ID" ]; then
    GPU_ID="0,1,2,3,4,5,6,7"
fi

python main.py \
    --dataset_name="RMT-team/babilong" \
    --model_path=$MODEL_PATH \
    --tag=$TAG \
    --adapter_path=$ADAPTER_PATH \
    --save_path='./results' \
    --gpu_id=$GPU_ID \
    --tp_size=1 \
    --test_full;


# bash scripts/run.sh "/mnt/petrelfs/tangzecheng/remote_bucket/zecheng/ckpt/pg19/Llama-3.1-8B/perturb_loss_3"
# bash scripts/run.sh "/mnt/petrelfs/tangzecheng/remote_bucket/zecheng/ckpt/pg19/Llama-3.1-8B/perturb_loss"
# bash scripts/run.sh "/mnt/petrelfs/tangzecheng/remote_bucket/zecheng/ckpt/pg19/Llama-3.1-8B/real_opposite_gradient_large_pos-1e-3-v2"
# bash scripts/run.sh "/mnt/hwfile/opendatalab/tangzecheng/local_ckpt/baseline/llama3.1-8B-pg19-longce/200ep"
# bash scripts/run.sh "/mnt/hwfile/opendatalab/tangzecheng/local_ckpt/pg19/Llama-3.1-8B/opposite_gradient_large_pos-1e-3-v2/global_step300"
# bash scripts/run.sh /mnt/hwfile/opendatalab/tangzecheng/local_ckpt/pg19/Llama-3.1-8B/cd_lm_full-0.005/global_step300 "" cd_lm_full-0.005