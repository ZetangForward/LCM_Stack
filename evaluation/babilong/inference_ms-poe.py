import os ,sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../custommodel/multiscale_transformer')))
from ms_poe_jbb import MsPoELlamaForCausalLM, setup_model,setup_tokenizer

import json,argparse,warnings
import numpy as np,pandas as pd
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from typing import List
import random

import multiprocessing as mp

import torch
import datasets

warnings.filterwarnings('ignore')

from utils.babilong.prompts import DEFAULT_PROMPTS, DEFAULT_TEMPLATE, get_formatted_input

# cd inference
# nohup python inference_ms-poe.py >ms_poe.log

BABILONG_DATA_PATH="/data/zecheng/Ms-PoE/babilong/"
RESULTS_DIR="/data/zecheng/acl2025/MyRLHF/evaluation/babilong/babilong_evals_ms_poe/"
TASKS=['qa1', 'qa2', 'qa3',  'qa4', 'qa5', 'qa6', 'qa7', 'qa8', 'qa9', 'qa10']
SPLIT_NAMES=['0k','1k','2k','4k','8k', '16k', '32k', '64k', '128k']
generate_kwargs = {
    'max_new_tokens': 20,
    'max_length': None,
    'num_beams': 1,
    'do_sample': False,
    'temperature': None,
    'top_p': None,
    'top_k': None,
    'pad_token_id': None
}

use_chat_template = True
use_instruction = True
use_examples = True
use_post_prompt = True



def allocate_gpu(tp_size=1,
                 gpu_list=list(range(torch.cuda.device_count()))):
    '''
    return :List[str] , each element is a group of gpus visable to a single process 
    '''
    gpu_list=list(map(str,gpu_list))
    if tp_size==1:return gpu_list
    return list(map(lambda x:", ".join(x),
                    [gpu_list[i:i+tp_size] for i in range(0,len(gpu_list),tp_size)]))


def chunks(lst, chunk_num):

    chunk_size=(len(lst)-1)//chunk_num + 1

    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i : i + chunk_size]


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



def prepare_babilong_data(
        tokenizer=None,
        data_dir=BABILONG_DATA_PATH,
        results_folder=RESULTS_DIR,
        tasks=TASKS,
        split_names=SPLIT_NAMES
        ):
    
    all_samples=[]

    for task in tqdm(tasks, desc='all'):
        # configure the prompt
        prompt_cfg = {
            'instruction': DEFAULT_PROMPTS[task]['instruction'] if use_instruction else '',
            'examples': DEFAULT_PROMPTS[task]['examples'] if use_examples else '',
            'post_prompt': DEFAULT_PROMPTS[task]['post_prompt'] if use_post_prompt else '',
            'template': DEFAULT_TEMPLATE,
            'chat_template': use_chat_template,
        }
        prompt_name = [f'{k}_yes' if prompt_cfg[k] else f'{k}_no' for k in prompt_cfg if k != 'template']
        prompt_name = '_'.join(prompt_name)

        for split_name in tqdm(split_names, desc=f'tasks: {task}'):
            # load dataset
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
                    # model_inputs = tokenizer.apply_chat_template(input_text, add_generation_prompt=True,
                    #                                              return_tensors='pt').cpu()
                    # model_inputs = {'input_ids': model_inputs}
                else:
                    raise ValueError("not_use_chat_template")
                    # model_inputs = tokenizer(input_text, return_tensors='pt',
                    #                          add_special_tokens=True).cpu()
                
                all_samples+=[{
                    # "model_inputs": model_inputs,
                    "input_text"  : input_text,
                    "target"      : target,
                    "question"    : question,
                    "task"        : task,
                    "split_name"  : split_name,
                    "index"       : index
                    

                }]
    random.shuffle(all_samples)
    return all_samples

def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


class MsPoEWorker(mp.Process):
    def __init__(self,
                 samples,
                 visable_gpu,
                 args,
                 results):
        super().__init__()
        self.samples=samples
        self.visable_gpu=visable_gpu
        self.args=args
        self.results=results
        os.environ['CUDA_VISIBLE_DEVICES'] = self.visable_gpu

    def run(self):
        model=setup_model(self.args)
        tokenizer=setup_tokenizer()

        for sample in tqdm(self.samples,desc=f"Process  {os.getpid()}: samples"):
            # model_inputs=sample['model_inputs']
            # model_inputs['input_ids']=model_inputs['input_ids'].to(model.device)
            input_text=sample['input_text']

            model_inputs = tokenizer.apply_chat_template(input_text, add_generation_prompt=True,
                                                                 return_tensors='pt').to(model.device)
            model_inputs = {'input_ids': model_inputs}
            
            sample_length = model_inputs['input_ids'].shape[1]
            with torch.no_grad():
                output = model.generate(**model_inputs, **generate_kwargs)
            output = output[0][sample_length:]
            output = tokenizer.decode(output, skip_special_tokens=True).strip()
            sample['output']=output
            # sample.pop('model_inputs')
            sample.pop("input_text")
            self.results+=[sample]

if __name__=="__main__":
    print("INTO-FILE")

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default="/data/zecheng/hf_models/Meta-Llama-3.1-8B-Instruct/")
    parser.add_argument("--saved_model_name", type=str, default="Llama3.1-8B-Instruct_head_metric_false")
    
    parser.add_argument("--results_folder",type=str,default=RESULTS_DIR)
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--apply_layers", type=str,
                         default="2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31")
    parser.add_argument("--head_type", type=str, default="normal")
    parser.add_argument("--enable_head_metrics", type=bool, default=False)
    parser.add_argument("--compress_ratio_min", type=float, default=1.2)
    parser.add_argument("--compress_ratio_max", type=float, default=1.8)
    parser.add_argument("--task_id",type=int, default=0)
    parser.add_argument("--n_gpu",type=int, default=torch.cuda.device_count())
    
    args = parser.parse_args()

    set_seed(args)

    mp.set_start_method("spawn")

    tokenizer= setup_tokenizer()
    if generate_kwargs['pad_token_id'] is None:
        generate_kwargs['pad_token_id'] = tokenizer.eos_token_id


    allocated_gpus=allocate_gpu(tp_size=2, gpu_list=[0,1,2,3,4,5,6,7])
    all_samples=prepare_babilong_data()
    prompts_chunks=chunks(all_samples, chunk_num=len(allocated_gpus))

    all_results=mp.Manager().list()
    processes = []
    for i,prompts_chunk in enumerate(prompts_chunks):
        process=MsPoEWorker(samples=prompts_chunk,
                            visable_gpu=allocated_gpus[i],
                            args=args,
                            results=all_results)
        processes.append(process)
        process.start()
    
    for process in processes:
        process.join()

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

        