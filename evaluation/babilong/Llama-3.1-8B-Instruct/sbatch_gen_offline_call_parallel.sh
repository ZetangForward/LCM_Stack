#!/bin/bash

#SBATCH --job-name=llama3.1-8B-Instruct-tool-call-parallel
#SBATCH --nodes=1                         
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=64        
#SBATCH --mem=400G                
#SBATCH --gres=gpu:8                       
#SBATCH --time=14-00:00:00 
#SBATCH --output=/mnt/petrelfs/tangzecheng/sbatch_logs/retrieval_then_gen_inf/job_id-%J.out       
#SBATCH --error=/mnt/petrelfs/tangzecheng/sbatch_logs/retrieval_then_gen_inf/job_id-%J.err         
#SBATCH --partition=belt_road
#SBATCH --exclusive             

export http_proxy=http://tangzecheng:Jn7iXe92XJUVYa5whNh07VJKZR6miGQ62it3goTiLBxRs8uZxkFD3gF0cQ3w@10.1.20.50:23128/ 
export https_proxy=http://tangzecheng:Jn7iXe92XJUVYa5whNh07VJKZR6miGQ62it3goTiLBxRs8uZxkFD3gF0cQ3w@10.1.20.50:23128/
export HTTP_PROXY=http://tangzecheng:Jn7iXe92XJUVYa5whNh07VJKZR6miGQ62it3goTiLBxRs8uZxkFD3gF0cQ3w@10.1.20.50:23128/
export HTTPS_PROXY=http://tangzecheng:Jn7iXe92XJUVYa5whNh07VJKZR6miGQ62it3goTiLBxRs8uZxkFD3gF0cQ3w@10.1.20.50:23128/

source /mnt/petrelfs/tangzecheng/anaconda3/etc/profile.d/conda.sh

conda activate zecheng_new

cd /mnt/petrelfs/tangzecheng/MyRLHF/inference

bash gen_offline.sh rapid_parallel_api tool_calling meta-llama/Meta-Llama-3.1-8B-Instruct /mnt/petrelfs/tangzecheng/local_data/first_retrieval_res/llama-3_1-8B-Instruct /mnt/petrelfs/tangzecheng/local_data/inference_results/llama-3_1-8B-Instruct/retrieval_then_gen api 8 2