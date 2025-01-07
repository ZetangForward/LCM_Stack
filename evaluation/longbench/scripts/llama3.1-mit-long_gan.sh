export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

python main.py \
    --model_path="/mnt/petrelfs/tangzecheng/local_ckpt/merge_v1_fix/Llama-3.1-8B-Instruct/gan/convert_global_step225" \
    --model_max_length_setting="normal_setting" \
    --tag="global_step_225" \
    --tp_size 4 \
    --gpu_lst="0,1,2,3,4,5,6,7" \
    --save_path="./results/Llama3.1/merge_v1_gan";

# echo "generate finish ..., begin to evaluate ..."

# python eval_longbench.py --pred_path=${SAVE_PATH}/${MODEL_NAME}
 