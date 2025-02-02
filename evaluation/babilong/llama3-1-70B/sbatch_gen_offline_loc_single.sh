#!/bin/bash

#SBATCH --job-name=llama-3_1-70B-tool-loc-single
#SBATCH --nodes=1                         
#SBATCH --ntasks-per-node=32 
#SBATCH --cpus-per-task=4        # 添加CPU配置
#SBATCH --mem=400G                # 添加内存配置
#SBATCH --gres=gpu:8                       
#SBATCH --time=14-00:00:00       # 设置具体的时间限制，比如14天     
#SBATCH --output=/mnt/petrelfs/tangzecheng/sbatch_logs/zc_job_id-%J.out       
#SBATCH --error=/mnt/petrelfs/tangzecheng/sbatch_logs/zc_job_id-%J.err         
#SBATCH --partition=belt_road   
#SBATCH --exclusive   

export http_proxy=http://tangzecheng:Jn7iXe92XJUVYa5whNh07VJKZR6miGQ62it3goTiLBxRs8uZxkFD3gF0cQ3w@10.1.20.50:23128/ 
export https_proxy=http://tangzecheng:Jn7iXe92XJUVYa5whNh07VJKZR6miGQ62it3goTiLBxRs8uZxkFD3gF0cQ3w@10.1.20.50:23128/
export HTTP_PROXY=http://tangzecheng:Jn7iXe92XJUVYa5whNh07VJKZR6miGQ62it3goTiLBxRs8uZxkFD3gF0cQ3w@10.1.20.50:23128/
export HTTPS_PROXY=http://tangzecheng:Jn7iXe92XJUVYa5whNh07VJKZR6miGQ62it3goTiLBxRs8uZxkFD3gF0cQ3w@10.1.20.50:23128/

source /mnt/petrelfs/tangzecheng/anaconda3/etc/profile.d/conda.sh

conda activate zecheng_new

cd /mnt/petrelfs/tangzecheng/MyRLHF/inference

bash gen_offline.sh rapid_single_api tool_location meta-llama/Meta-Llama-3.1-8B-Instruct /mnt/hwfile/opendatalab/tangzecheng/benchmark_data /mnt/petrelfs/tangzecheng/local_data/inference_results/llama-3_1-8B-Instruct api 8 1