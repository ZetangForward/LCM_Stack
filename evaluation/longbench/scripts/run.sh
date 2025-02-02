MODEL_PATH=$1
ADAPTER_PATH=$2
TAG=$3
GPU_LST=$4
MODEL_CONFIG=$5
MAX_LENGTH_SETTING=$6

if [ -z "$ADAPTER_PATH" ]; then
    ADAPTER_PATH=""
fi

if [ -z "$TAG" ]; then
    TAG=""
fi

if [ -z "$GPU_LST" ]; then
    GPU_LST="0,1,2,3,4,5,6,7"
fi

if [ -z "$MODEL_CONFIG" ]; then
    MODEL_CONFIG=""
fi

if [ -z "$MAX_LENGTH_SETTING" ]; then
    MAX_LENGTH_SETTING="normal_setting"
fi

python main.py \
    --model_path=$MODEL_PATH \
    --model_config=$MODEL_CONFIG \
    --adapter_path=$ADAPTER_PATH \
    --gpu_lst=$GPU_LST \
    --tag=$TAG \
    --model_max_length_setting="normal_setting" \
    --save_path="./results";


# bash scripts/contextual_denoising/llama3.1-8b-base.sh "/mnt/petrelfs/tangzecheng/remote_bucket/zecheng/ckpt/pg19/Llama-3.1-8B/perturb_loss_3"
# bash scripts/contextual_denoising/llama3.1-8b-base.sh "/mnt/petrelfs/tangzecheng/remote_bucket/zecheng/ckpt/pg19/Llama-3.1-8B/perturb_loss"
# bash scripts/contextual_denoising/llama3.1-8b-base.sh "/mnt/petrelfs/tangzecheng/remote_bucket/zecheng/ckpt/pg19/Llama-3.1-8B/real_opposite_gradient_large_pos-1e-3-v2"