#!/bin/bash

#SBATCH --job-name=inference_babilong
#SBATCH --nodes=1                         
#SBATCH --ntasks-per-node=4 
#SBATCH --cpus-per-task=4        
#SBATCH --mem=200G               
#SBATCH --gres=gpu:4                       
#SBATCH --time=14-00:00:00         
#SBATCH --output=/mnt/petrelfs/tangzecheng/LCM_Stack/evaluation/longbench/logs/%J.out       
#SBATCH --error=/mnt/petrelfs/tangzecheng/LCM_Stack/evaluation/longbench/logs/%J.err         
#SBATCH --partition=belt_road     

unset http_proxy; unset https_proxy; unset HTTP_PROXY; unset HTTPS_PROXY

source /mnt/petrelfs/tangzecheng/anaconda3/etc/profile.d/conda.sh

conda activate zecheng

mkdir -p /mnt/petrelfs/tangzecheng/remote_bucket

/mnt/petrelfs/tangzecheng/goofys \
    --endpoint http://10.140.31.254:80 \
    --debug_s3 \
    wulijun_blob /mnt/petrelfs/tangzecheng/remote_bucket


export http_proxy=http://tangzecheng:Jn7iXe92XJUVYa5whNh07VJKZR6miGQ62it3goTiLBxRs8uZxkFD3gF0cQ3w@10.1.20.50:23128/ 
export https_proxy=http://tangzecheng:Jn7iXe92XJUVYa5whNh07VJKZR6miGQ62it3goTiLBxRs8uZxkFD3gF0cQ3w@10.1.20.50:23128/
export HTTP_PROXY=http://tangzecheng:Jn7iXe92XJUVYa5whNh07VJKZR6miGQ62it3goTiLBxRs8uZxkFD3gF0cQ3w@10.1.20.50:23128/
export HTTPS_PROXY=http://tangzecheng:Jn7iXe92XJUVYa5whNh07VJKZR6miGQ62it3goTiLBxRs8uZxkFD3gF0cQ3w@10.1.20.50:23128/

cd /mnt/petrelfs/tangzecheng/LCM_Stack/evaluation/longbench

bash scripts/run.sh "/mnt/petrelfs/tangzecheng/remote_bucket/zecheng/ckpt/pg19/Llama-3-8B-Scaling-Noise/full_v2/global_step450" "" "" "Llama-3-8B-Scaling-Noise/full_v2/global_step450"