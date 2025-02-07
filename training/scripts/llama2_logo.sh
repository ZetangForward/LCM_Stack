ZERO1="./config/zero1.json"
ZERO2="./config/zero2.json"
ZERO3="./config/zero3.json"
FAST_ZERO3="./config/zero3-fast.json"
DIR_PATH=$1
SAVE_DIR=$2 
MODEL_DIR=$3

deepspeed --include localhost:0,1,2,3,4,5,6,7 ./logo_train.py \
    --output_dir=${SAVE_DIR}/llama2_logo \
    --max_steps=1000 \
    --logging_steps=1 \
    --per_device_train_batch_size=1 \
    --per_device_eval_batch_size=1 \
    --gradient_accumulation_steps=8 \
    --gradient_checkpointing=True \
    --learning_rate=5e-7 \
    --model_name_or_path=${MODEL_DIR} \
    --lr_scheduler_type="cosine" \
    --max_position_embeddings=81920 \
    --warmup_steps=100 \
    --save_steps=50 \
    --lora_r=32 \
    --max_target_length=2000 \
    --lora_alpha=16 \
    --max_seq_length=18000 \
    --dataloader_num_workers=0 \
    --save_total_limit=10 \
    --save_strategy="steps" \
    --eval_strategy="steps" \
    --eval_steps=100 \
    --load_best_model_at_end False \
    --weight_decay=0.05 \
    --optim="paged_adamw_32bit" \
    --report_to="tensorboard" \
    --dataset_path=${DATA_DIR} \
    --low_rank_training True \
    --remove_unused_columns False \
    --deepspeed=${ZERO2} \
    --bf16 True \
    --beta 3 \
    --sft_weight 0.1 \
    --gamma_beta_ratio=0.2;