#!/bin/bash

#SBATCH --job-name=inference_babilong
#SBATCH --nodes=1                         
#SBATCH --ntasks-per-node=4 
#SBATCH --cpus-per-task=4        
#SBATCH --mem=100G               
#SBATCH --gres=gpu:4                       
#SBATCH --time=14-00:00:00         
#SBATCH --output=/mnt/petrelfs/tangzecheng/LCM_Stack/evaluation/longbench/logs/%J.out       
#SBATCH --error=/mnt/petrelfs/tangzecheng/LCM_Stack/evaluation/longbench/logs/%J.err         
#SBATCH --partition=belt_road     

export http_proxy=http://tangzecheng:Jn7iXe92XJUVYa5whNh07VJKZR6miGQ62it3goTiLBxRs8uZxkFD3gF0cQ3w@10.1.20.50:23128/ 
export https_proxy=http://tangzecheng:Jn7iXe92XJUVYa5whNh07VJKZR6miGQ62it3goTiLBxRs8uZxkFD3gF0cQ3w@10.1.20.50:23128/
export HTTP_PROXY=http://tangzecheng:Jn7iXe92XJUVYa5whNh07VJKZR6miGQ62it3goTiLBxRs8uZxkFD3gF0cQ3w@10.1.20.50:23128/
export HTTPS_PROXY=http://tangzecheng:Jn7iXe92XJUVYa5whNh07VJKZR6miGQ62it3goTiLBxRs8uZxkFD3gF0cQ3w@10.1.20.50:23128/

source /mnt/petrelfs/tangzecheng/anaconda3/etc/profile.d/conda.sh

conda activate zecheng

cd /mnt/petrelfs/tangzecheng/LCM_Stack/evaluation/longbench

bash scripts/run.sh "meta-llama/Meta-Llama-3.1-8B-Instruct" "/mnt/petrelfs/tangzecheng/remote_bucket/zecheng/ckpt/baseline/Llama-3.1-8B-Instruct/longalpaca/adapter/global_step250" "Llama-3.1-8B-Instruct-longalpaca"