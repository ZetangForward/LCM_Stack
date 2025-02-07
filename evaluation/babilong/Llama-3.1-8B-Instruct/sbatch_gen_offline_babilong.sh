#!/bin/bash

#SBATCH --job-name=llama3.1-8B-Instruct-babilong
#SBATCH --nodes=1                         
#SBATCH --ntasks-per-node=4 
#SBATCH --cpus-per-task=12        # 添加CPU配置
#SBATCH --mem=400G                # 添加内存配置
#SBATCH --gres=gpu:8                       
#SBATCH --time=14-00:00:00       # 设置具体的时间限制，比如14天     
#SBATCH --output=/mnt/petrelfs/tangzecheng/sbatch_logs/zc_job_id-%J.out       
#SBATCH --error=/mnt/petrelfs/tangzecheng/sbatch_logs/zc_job_id-%J.err         
#SBATCH --partition=belt_road        
#SBATCH --quotatype=spot     

export http_proxy=http://tangzecheng:Jn7iXe92XJUVYa5whNh07VJKZR6miGQ62it3goTiLBxRs8uZxkFD3gF0cQ3w@10.1.20.50:23128/ 
export https_proxy=http://tangzecheng:Jn7iXe92XJUVYa5whNh07VJKZR6miGQ62it3goTiLBxRs8uZxkFD3gF0cQ3w@10.1.20.50:23128/
export HTTP_PROXY=http://tangzecheng:Jn7iXe92XJUVYa5whNh07VJKZR6miGQ62it3goTiLBxRs8uZxkFD3gF0cQ3w@10.1.20.50:23128/
export HTTPS_PROXY=http://tangzecheng:Jn7iXe92XJUVYa5whNh07VJKZR6miGQ62it3goTiLBxRs8uZxkFD3gF0cQ3w@10.1.20.50:23128/

source /mnt/petrelfs/tangzecheng/anaconda3/etc/profile.d/conda.sh

conda activate zecheng

cd /mnt/petrelfs/tangzecheng/MyRLHF/inference

python inference_babilong.py \
    --dataset_name "RMT-team/babilong" \
    --model_path 'meta-llama/Meta-Llama-3.1-8B-Instruct' \
    --adapter_path '/mnt/petrelfs/tangzecheng/local_ckpt/merge_v1/Llama-3.1-8B-Instruct/simpo/global_step300' \
    --save_path '/mnt/petrelfs/tangzecheng/MyRLHF/inference/babilong/merge_v1/Llama-3.1-8B-Instruct/' \
    --num_gpus 4 \
    --tp_size 1 \

# bash gen_offline.sh babilong reasoning meta-llama/Meta-Llama-3.1-8B-Instruct "RMT-team/babilong" /mnt/petrelfs/tangzecheng/local_data/inference_results/llama-3_1-8B-Instruct babilong 8 2