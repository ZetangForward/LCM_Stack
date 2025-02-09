from modelzipper.tutils import *
from utils import ALL_LB_TESTING_SETS, LB_DATA_PROMPT, LB_PRED_LEN, DATASET2CATEGORY, LB_DATA_PROMPT_TEMPLATE
from datasets import load_dataset
import argparse
import torch
import torch.multiprocessing as mp
import numpy as np
from loguru import logger
from peft import PeftModelForCausalLM
import os
import openai
import time
from transformers import AutoConfig

context_max_length = {
    "8k_setting": 7200, 
    "tiny_setting": 15500, 
    "normal_setting": 32000, 
    "long_setting": 63500, 
    "ultra_long_setting": 127500
}

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



def query_gpt4(datasets, dataset_name, return_list):
    openai.api_key = "sk-8BG3gQz68h89HLdz04D54a4dF41845D49548F0293bDe5c44" # sk-47vSsxMRgigQ0SanDd79E52e95De4bF48b8c56F5B5797c2d
    openai.base_url = 'https://chatapi.onechats.top/v1/'  # https://api.onechats.cn/v1/
    # openai.api_key = "sk-47vSsxMRgigQ0SanDd79E52e95De4bF48b8c56F5B5797c2d"
    # openai.base_url = 'https://api.onechats.cn/v1/'
    PROMPT_TEMPLATE, PROMPT_CHAT_TEMPLATE, max_new_token = LB_DATA_PROMPT[dataset_name], LB_DATA_PROMPT_TEMPLATE[dataset_name], LB_PRED_LEN[dataset_name]
    pred_res = []
    print(f"start to process {dataset_name} ...")
    with tqdm(total=len(datasets)) as pbar:
        for sample in datasets:
            context, input_, answer = sample['context'], sample['input'], sample['answers']
            prompt = PROMPT_TEMPLATE.format(input=input_, context=context)
            if  "length" in sample:
                length = sample["length"]
            else:
                length = 0
            try_times = 0
            while try_times < 3:
                try:
                    response = openai.chat.completions.create(
                        model="gpt-4o-2024-11-20",
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=max_new_token,
                        temperature=0
                    )
                    pred_str = response.choices[0].message.content
                    pred_res.append({"dataset_name": dataset_name, "pred_str": pred_str, "answers": answer, "length": length})
                    break
                except Exception as e:
                    print(f"发生错误: {str(e)}")
                    print("3秒后重试...")
                    time.sleep(3)
                    try_times += 1
            
            pbar.update(1)

    return_list.extend(pred_res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="lb testing")
    parser.add_argument('--model_path', type=str, default=None, help='Path to the model')
    parser.add_argument('--adapter_path', type=str, default=None, help='Path to the PEFT model')
    parser.add_argument('--save_path', type=str, default=None, help='Path to save the output')
    parser.add_argument('--gpu_lst', type=str, default=None, help='All available gpus')
    parser.add_argument('--tp_size', type=int, default=1, help='model parallel size')
    parser.add_argument('--tag', type=str, default=None, help='output_dir tag')
    parser.add_argument('--chat_template', type=str, default=None, help='chat template')
    parser.add_argument('--model_max_length_setting', type=str, default="normal_setting", help='Model max length setting')
    parser.add_argument('--seed', type=int, default=27, help='default seed')
    parser.add_argument('--model_config', type=str, default=None, help='model config')

    args = parser.parse_args()
    
    mp.set_start_method('spawn', force=True)

    world_size = 8

    pred_dir = "/data/zecheng/LCM_Stack/evaluation/longbench/results/gpt4o"
    
    if os.path.exists(pred_dir):
        already_finish_files = auto_read_dir(pred_dir, file_suffix=".jsonl")
        already_finish_files = [os.path.basename(f).split('.')[0] for f in already_finish_files]
        
        # check generated cases
        for f in already_finish_files[::-1]:
            num_test_cases = len(load_dataset('THUDM/LongBench', f, split='test', trust_remote_code=True))
            num_pred_cases = len(auto_read_data(os.path.join(pred_dir, f + ".jsonl")))
            if num_test_cases != num_pred_cases: 
                print(f"{f} has not been processed, removing it from finished files ...")
                already_finish_files.remove(f)
    else:
        auto_mkdir(pred_dir)
        already_finish_files = []
    
    test_datasets = list(set(ALL_LB_TESTING_SETS) - set(already_finish_files))
    logger.info(f"evaluating on datasets: {test_datasets}")
    max_context_length = context_max_length[args.model_max_length_setting]
    logger.info(f"max_context_length are set as {max_context_length}")
    
    for dataset_name in test_datasets:
        test_data = load_dataset('THUDM/LongBench', dataset_name, split='test', trust_remote_code=True)
        save_res_path = os.path.join(pred_dir, dataset_name + ".jsonl")
        data_subsets = []
        
        for i in range(world_size):
            subset = test_data.shard(num_shards=world_size, index=i)
            data_subsets.append(subset)

        with tqdm(total=world_size) as pbar:
            processes = []
            manager = mp.Manager()
            return_list = manager.list()
            
            for rank in range(0, world_size):
                p = mp.Process(target=query_gpt4, args=(data_subsets[rank], dataset_name, return_list))
                p.start()
                processes.append(p)
            
            for p in processes:
                p.join()
                pbar.update()
        
        results = list(return_list)

        auto_save_data(results, save_res_path)

    # 最后一起进行评测
    logger.info("start to eval ...")
    # 定义命令字符串
    command = f'python eval.py --pred_path={pred_dir}'

    # 执行命令
    exit_code = os.system(command)

    # 检查命令是否成功执行
    if exit_code == 0:
        print("命令执行成功！")
    else:
        print(f"命令执行失败，退出码：{exit_code}")