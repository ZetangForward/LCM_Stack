import argparse
import json
import multiprocessing as mp
import os
import sys
import numpy as np
import copy
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from modelzipper.tutils import *
from datasets import load_dataset
from utils.babilong.prompts import DEFAULT_PROMPTS, DEFAULT_TEMPLATE, get_formatted_input

BABILONG_SYSTEM_PROMPT = """You are an AI assistant that explains your reasoning step by step, incorporating dynamic Chain of Thought (CoT). Follow these instructions:\n\n1. Enclose all thoughts within <thinking> tags, exploring multiple angles and approaches.\n2. Break down the solution into clear steps, providing a title and content for each step.\n3. After each step, decide if you need another step or if you're ready to give the final answer.\n4. Explore multiple solutions individually if possible, comparing approaches in your reflections.\n5. Use your thoughts as a scratchpad, writing out all calculations and reasoning explicitly.\n\nYour goal is to demonstrate a thorough, adaptive, and self-reflective problem-solving process, emphasizing dynamic thinking and learning from your own reasoning."""

inference_args = dict(
    top_p = dict(
        n = 1, 
        temperature = 0.7, 
        max_tokens = 100, 
        seed = 42, 
        top_p = 0.95,
    ),
    top_n = dict(
        n = 6, 
        temperature = 0.7, 
        max_tokens = 100, 
        seed = 42, 
        top_p = 0.95,
    ),
    greedy = dict(
         n = 1,
        temperature = 0.0,
        max_tokens = 100,
        seed = 42,
    )
)


def prepare_babilong_data(data_dir, tokenizer, inference_scaling=False):
    tasks = ['qa2', 'qa3', 'qa4', 'qa5', 'qa6', 'qa7']
    split_names = ['0k', '1k', '2k', '4k', '8k', '16k', '32k', '64k']  # 目前先不测试到 128K
    all_input_texts = []

    for task in tqdm(tasks, desc='tasks'):
        # configure the prompt
        """ zecheng_note
        meta_data = {
            "ids": 0, 
            "task: xxx, 
            "subtask": 
        }
        prefix_prompt / system_prompt
        context
        question
        template
        ** postprocessing.py ->  
        golden
        """
        prompt_cfg = {
            'instruction': DEFAULT_PROMPTS[task]['instruction'],
            'examples': DEFAULT_PROMPTS[task]['examples'],
            'post_prompt': DEFAULT_PROMPTS[task]['post_prompt'],
            'template': DEFAULT_TEMPLATE,
            'chat_template': True,
        }
        prompt_name = [f'{k}_yes' if prompt_cfg[k] else f'{k}_no' for k in prompt_cfg if k != 'template']
        prompt_name = '_'.join(prompt_name)
        for split_name in tqdm(split_names, desc='lengths'):
            # load dataset
            data = load_dataset(data_dir, split_name, cache_dir="/mnt/petrelfs/tangzecheng/local_data/cache")
            task_data = data[task]
            for sample in tqdm(task_data, desc=f'task: {task} length: {split_name}'):
                target, context, question = sample['target'], sample['input'], sample['question']
                input_text = get_formatted_input(
                    context, question, prompt_cfg['examples'],
                    prompt_cfg['instruction'], prompt_cfg['post_prompt'],
                    template=prompt_cfg['template']
                )
                if inference_scaling:
                    model_inputs = tokenizer.apply_chat_template(
                        [{'role': 'system', 'content': BABILONG_SYSTEM_PROMPT},
                        {'role': 'user', 'content': input_text}], 
                        add_generation_prompt=True, tokenize=False
                    )
                else:
                    model_inputs = tokenizer.apply_chat_template(
                        [{'role': 'user', 'content': input_text}], 
                        add_generation_prompt=True, tokenize=False
                    )
                all_input_texts.append({"message": model_inputs, "golden": target, "task": task, "ctx_length": split_name, 'question': question})
    return all_input_texts             


def worker(gpu_ids: str, prompts_chunk, model_path, model_args, inference_args, return_list):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids

    llm = LLM(model=model_path, **model_args)
    sampling_params = SamplingParams(**inference_args)
    chunk_message = [item['message'] for item in prompts_chunk]

    logger.info(f"Start to generate {len(chunk_message)} samples")

    outputs = llm.generate(chunk_message, sampling_params=sampling_params, use_tqdm=True)

    results = []

    for i, output in enumerate(outputs):
        generations = [o.text for o in output.outputs]
        original_prompt_data = prompts_chunk[i]  # 获取原始数据
        original_prompt_data['pred'] = generations  # 插入生成的响应
        original_prompt_data.pop('message')
        results.append(original_prompt_data)

    return_list.extend(results)

def main():
    parser = argparse.ArgumentParser(description="Inference with VLLM")
    parser.add_argument('--dataset_name', type=str, default=None, help='Name of the dataset')
    parser.add_argument('--data_dir', type=str, default=None, help='Path to the data directory')
    parser.add_argument('--dialogue_turn', type=int, default=1, help='Turn of the dialogue')
    parser.add_argument('--benchmark_name', type=str, default=None, help='Name of the benchmark')
    parser.add_argument('--task_name', type=str, default=None, help='Name of the task')
    parser.add_argument('--model_path', type=str, default=None, help='Path to the model')
    parser.add_argument('--peft_path', type=str, default=None, help='Path to the PEFT model')
    parser.add_argument('--save_path', type=str, default=None, help='Path to save the output')
    parser.add_argument('--seed', type=int, default=27, help='Default seed value')
    parser.add_argument('--max_model_len', type=int, default=64000, help='model max context length')
    parser.add_argument('--max_workers', type=int, default=2, help='Maximum number of worker threads')
    parser.add_argument('--use_logn', action='store_true', help='Flag to use log-normal distribution')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature parameter for generation')
    parser.add_argument('--k', type=int, default=1, help='Number of generations per prompt')
    parser.add_argument('--num_gpus', type=int, default=8, help='Number of GPUs to use')
    parser.add_argument('--tp_size', type=int, default=1, help='Tensor parallel size')
    args = parser.parse_args()

    assert args.save_path is not None, "save_path is not set"
    
    auto_mkdir(args.save_path)

    torch.cuda.manual_seed_all(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    input_queries = prepare_babilong_data(args.data_dir, tokenizer)
    
    out_file_path = os.path.join(args.save_path, f"preds_{args.dataset_name}.jsonl")

    model_args = {
        "tensor_parallel_size": args.tp_size, 
        "gpu_memory_utilization": 0.98,
        "swap_space": 12,
        "max_model_len": args.max_model_len, 
        "trust_remote_code": True, 
    }

    chunk_num = args.num_gpus // args.tp_size
    chunk_size = (len(input_queries) + chunk_num - 1) // chunk_num
    prompts_chunks = [input_queries[i*chunk_size:(i+1)*chunk_size] for i in range(chunk_num)]
  
    manager = mp.Manager()
    return_list = manager.list()
    processes = []

    # construct gpu_ids list
    if args.tp_size == 1:
        gpu_id_lst = [str(i) for i in range(args.num_gpus)]
    else:
        gpu_id_lst = []

        for i in range(0, args.num_gpus, args.tp_size):
            tmp = list(range(i, i + args.tp_size))
            gpu_id_lst.append(", ".join([str(i) for i in tmp]))
    
    # worker(gpu_ids, prompts_chunks[0], args.model_path, model_args, inference_args['top_p'], return_list)
    
    # 使用 tqdm 显示总进度
    for chunk_id, gpu_ids in enumerate(gpu_id_lst):
        p = mp.Process(target=worker, args=(gpu_ids, prompts_chunks[chunk_id], args.model_path, model_args, inference_args['top_p'], return_list))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    
    # 保存生成结果
    logger.info('Have collected ', len(return_list), 'samples, begin to save ...')
    auto_save_data(return_list, out_file_path)

if __name__ == '__main__':
    main()
    # prepare_babilong_data("")
