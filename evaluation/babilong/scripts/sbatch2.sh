#!/bin/bash

#SBATCH --job-name=inference_babilong
#SBATCH --nodes=1                         
#SBATCH --ntasks-per-node=8 
#SBATCH --cpus-per-task=16        # 添加CPU配置
#SBATCH --mem=400G                # 添加内存配置
#SBATCH --gres=gpu:8                       
#SBATCH --time=14-00:00:00       # 设置具体的时间限制，比如14天     
#SBATCH --output=/mnt/petrelfs/tangzecheng/sbatch_logs/%J.out       
#SBATCH --error=/mnt/petrelfs/tangzecheng/sbatch_logs/%J.err         
#SBATCH --partition=belt_road     
#SBATCH --exclusive     

export http_proxy=http://tangzecheng:Jn7iXe92XJUVYa5whNh07VJKZR6miGQ62it3goTiLBxRs8uZxkFD3gF0cQ3w@10.1.20.50:23128/ 
export https_proxy=http://tangzecheng:Jn7iXe92XJUVYa5whNh07VJKZR6miGQ62it3goTiLBxRs8uZxkFD3gF0cQ3w@10.1.20.50:23128/
export HTTP_PROXY=http://tangzecheng:Jn7iXe92XJUVYa5whNh07VJKZR6miGQ62it3goTiLBxRs8uZxkFD3gF0cQ3w@10.1.20.50:23128/
export HTTPS_PROXY=http://tangzecheng:Jn7iXe92XJUVYa5whNh07VJKZR6miGQ62it3goTiLBxRs8uZxkFD3gF0cQ3w@10.1.20.50:23128/

source /mnt/petrelfs/tangzecheng/anaconda3/etc/profile.d/conda.sh

conda activate zecheng

cd /mnt/petrelfs/tangzecheng/LCM_Stack/evaluation/babilong

bash scripts/run.sh "/mnt/hwfile/opendatalab/tangzecheng/local_ckpt/baseline/llama3.1-8B-pg19-ce/global_step150_hf"