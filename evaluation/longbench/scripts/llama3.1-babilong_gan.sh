export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python main.py \
    --model_path="/mnt/petrelfs/tangzecheng/local_ckpt/babilong/Llama-3.1-8B-Instruct/gan/convert_step100" \
    --model_max_length_setting="normal_setting" \
    --tag="global_step_100" \
    --save_path="./results/Llama3.1/gan";

# echo "generate finish ..., begin to evaluate ..."

# python eval_longbench.py --pred_path=${SAVE_PATH}/${MODEL_NAME}
 