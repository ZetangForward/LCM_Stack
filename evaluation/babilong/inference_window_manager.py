import os ,sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../custommodel/multiscale_transformer')))
from test_jbb import setup_model,setup_tokenizer

import json,argparse,warnings
import numpy as np,pandas as pd
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from typing import List

import torch
import datasets
from argparse import Namespace

from utils.babilong.prompts import DEFAULT_PROMPTS, DEFAULT_TEMPLATE, get_formatted_input
from utils.mspoe.manager.gpumanager import LLMEvalGPUManager

BABILONG_DATA_PATH="/data/zecheng/Ms-PoE/babilong/"
RESULTS_DIR="/data/zecheng/acl2025/MyRLHF/evaluation/babilong/babilong_evals_ms_poe/"
TASKS=['qa1','qa2','qa3','qa4', 'qa5','qa6', 'qa7', 'qa8', 'qa9', 'qa10']
SPLIT_NAMES=['0k', '1k','2k','4k','8k', '16k', '32k', '64k', '128k']
use_chat_template = True
use_instruction = True
use_examples = True
use_post_prompt = True

# nohup python inference_window_manager.py --group_size 16 > extend_group16.log

class BabilongManager(LLMEvalGPUManager):
    @classmethod
    def setup_model(cls, model_config: Namespace):
        model=setup_model(model_config)
        return model
    @classmethod
    def preprocess(cls, task_info, data_config: Namespace, glb: Namespace):
        tokenizer = data_config.tokenizer
        data_dir=BABILONG_DATA_PATH
        
        task=task_info.task
        split_name=task_info.split_name

        prompt_cfg=task_info.prompt_cfg
        prompt_name = [f'{k}_yes' if prompt_cfg[k] else f'{k}_no' for k in prompt_cfg if k != 'template']
        prompt_name = '_'.join(prompt_name)

        data = datasets.load_dataset(data_dir, split_name)
        task_data = data[task]

        for index,sample in enumerate(task_data):
            target = sample['target']
            context = sample['input']
            question = sample['question']

            # format input text
            input_text = get_formatted_input(context, question, prompt_cfg['examples'],
                                            prompt_cfg['instruction'], prompt_cfg['post_prompt'],
                                            template=prompt_cfg['template'])

            if use_chat_template:
                input_text = [{'role': 'user', 'content': input_text}]
                model_inputs = tokenizer.apply_chat_template(input_text, add_generation_prompt=True,
                                                            return_tensors='pt').cpu()
                model_inputs = {'input_ids': model_inputs}
            else:
                raise ValueError("not_use_chat_template")
            
            yield {
                "model_inputs": model_inputs,
                "target"      : target,
                "question"    : question,
                "task"        : task,
                "split_name"  : split_name,
                "index"       : index
            }

    @classmethod
    def process(cls, model, sample, generate_config: Namespace):
        tokenizer=generate_config.tokenizer
        model_inputs=sample['model_inputs']
        model_inputs['input_ids']=model_inputs['input_ids'].to(model.device)

        sample_length = model_inputs['input_ids'].shape[1]
        with torch.no_grad():
            output = model.generate(**model_inputs, **vars(generate_config))
        output = output[0][sample_length:]
        output = tokenizer.decode(output, skip_special_tokens=True).strip()
        sample['output']=output
        sample.pop('model_inputs')
        # sample.pop('input_text')

        yield sample


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def list_to_df(lst: List[dict],sort_index=None):
    if len(lst)==0:
        print("`list_to_df`: No datas!")
        return None
    if sort_index is not None:
        lst.sort(key=lambda x: x[sort_index])
    df={k:[] for k in lst[0].keys()}
    for ele in lst:
        for k,v in ele.items():
            df[k]+=[v]
    return pd.DataFrame(df)
    


if __name__=="__main__":
    print("INTO-FILE")

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default="/data/zecheng/hf_models/Meta-Llama-3.1-8B-Instruct/")
    parser.add_argument("--saved_model_name", type=str, default="Llama3.1-8B-Instruct_extend")
    
    parser.add_argument("--results_folder",type=str,default=RESULTS_DIR)
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--task_id",type=int, default=0)
    parser.add_argument("--n_gpu",type=int, default=torch.cuda.device_count())
    parser.add_argument("--group_size",type=int,default=16)

    args = parser.parse_args()



    args.saved_model_name+="_group"+str(args.group_size)

    set_seed(args)

    tokenizer= setup_tokenizer()
    generate_kwargs = {
        'max_new_tokens': 20,
        'max_length': None,
        'num_beams': 1,
        'do_sample': False,
        'temperature': None,
        'top_p': None,
        'top_k': None,
        'pad_token_id': tokenizer.eos_token_id
    }

    data_config=Namespace(tokenizer=tokenizer,**vars(args))
    generate_config=Namespace(tokenizer=tokenizer,**generate_kwargs)

    task_info_list=[]
    for task in TASKS:
        prompt_cfg = {
            'instruction': DEFAULT_PROMPTS[task]['instruction'] if use_instruction else '',
            'examples': DEFAULT_PROMPTS[task]['examples'] if use_examples else '',
            'post_prompt': DEFAULT_PROMPTS[task]['post_prompt'] if use_post_prompt else '',
            'template': DEFAULT_TEMPLATE,
            'chat_template': use_chat_template,
        }
        for split_name in SPLIT_NAMES:
            task_info_list+=[Namespace(task=task,prompt_cfg=prompt_cfg,split_name=split_name)]


    with BabilongManager(
        tp_size=1,
        gpu_list=[0,1,2,3] if args.group_size==16 else [4,5,6,7],
        task_info_list=task_info_list,
        basic_datas_number=len(task_info_list)*900,
        data_config=data_config,
        model_config=args,
        generate_config=generate_config
    ) as manager:
        all_results=manager.result_list


    # save  results
    results_dict=defaultdict(lambda : defaultdict(list))
    for result in all_results:
        results_dict[result['task']][result['split_name']].append(result)

    for task in TASKS:
        prompt_cfg = {
            'instruction': DEFAULT_PROMPTS[task]['instruction'] if use_instruction else '',
            'examples': DEFAULT_PROMPTS[task]['examples'] if use_examples else '',
            'post_prompt': DEFAULT_PROMPTS[task]['post_prompt'] if use_post_prompt else '',
            'template': DEFAULT_TEMPLATE,
            'chat_template': use_chat_template,
        }
        prompt_name = [f'{k}_yes' if prompt_cfg[k] else f'{k}_no' for k in prompt_cfg if k != 'template']
        prompt_name = '_'.join(prompt_name)

    
        for split_name in tqdm(SPLIT_NAMES, desc='lengths'):

            outfile = Path(f'{args.results_folder}/{args.saved_model_name}/{task}_{split_name}_{prompt_name}.csv')
            outfile.parent.mkdir(parents=True, exist_ok=True)
            cfg_file = f'{args.results_folder}/{args.saved_model_name}/{task}_{split_name}_{prompt_name}.json'
            json.dump({'prompt': prompt_cfg, 'generate_kwargs': generate_kwargs}, open(cfg_file, 'w'), indent=4)

            df=list_to_df(results_dict[task][split_name],sort_index='index')
            if df is None:
                print("Missing datas: ",outfile)
                continue
            df=df[['target','output','question']]
            df.to_csv(outfile)

        