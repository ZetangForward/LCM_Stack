model_path=$1
out_dir=$2
input_data_dir=$3

ulimit -n 4096
python gen_hf.py \
    --ports 4100 4101 4102 4103 4104 4105 4106 4107 \
    --tokenizer $model_path \
    --input_data_dir $input_data_dir \
    --output_dir $out_dir --K 1 \
    --max_workers 16 \
    --temperature 0.7 \
    --model_type 'llama-3' \
    --strategy 'wrong';