import argparse
import json
import multiprocessing as mp
import os
import sys
import numpy as np
import copy
from tqdm import trange
from transformers import AutoTokenizer, AutoModelForCausalLM
from modelzipper.tutils import *
from peft import PeftModelForCausalLM
from datasets import load_dataset
from utils.babilong.prompts import DEFAULT_PROMPTS, DEFAULT_TEMPLATE, get_formatted_input, LongMIT_DEFAULT_TEMPLATE
import pynvml

# Initialize NVML (NVIDIA Management Library)
pynvml.nvmlInit()


BABILONG_SYSTEM_PROMPT = """You are an AI assistant that explains your reasoning step by step, incorporating dynamic Chain of Thought (CoT). Follow these instructions:\n\n1. Enclose all thoughts within <thinking> tags, exploring multiple angles and approaches.\n2. Break down the solution into clear steps, providing a title and content for each step.\n3. After each step, decide if you need another step or if you're ready to give the final answer.\n4. Explore multiple solutions individually if possible, comparing approaches in your reflections.\n5. Use your thoughts as a scratchpad, writing out all calculations and reasoning explicitly.\n\nYour goal is to demonstrate a thorough, adaptive, and self-reflective problem-solving process, emphasizing dynamic thinking and learning from your own reasoning."""

inference_args = dict(
    top_p = dict(
        temperature = 0.7, 
        max_new_tokens = 100, 
        do_sample = True, 
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


def get_available_gpus(max_memory_usage=200):
    """
    Detect available GPUs with memory usage less than the specified threshold.
    
    Args:
        max_memory_usage (int): Maximum memory usage (in MB) to consider a GPU as available.

    Returns:
        list: List of GPU IDs that are available.
    """
    available_gpus = []
    gpu_count = pynvml.nvmlDeviceGetCount()
    
    for i in range(gpu_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used_memory_mb = mem_info.used / (1024 * 1024)  # Convert bytes to MB
        
        if used_memory_mb < max_memory_usage:
            available_gpus.append(i)
    
    return available_gpus

# Construct gpu_ids list
def construct_gpu_ids(args):
    """
    Construct GPU IDs list based on tensor parallel size and available GPUs.

    Args:
        args: Namespace containing `tp_size` and `num_gpus` attributes.

    Returns:
        list: List of GPU ID strings.
    """
    # Detect available GPUs
    available_gpus = get_available_gpus()
    
    if args.tp_size == 1:
        gpu_id_lst = [str(i) for i in available_gpus]
    else:
        gpu_id_lst = []
        for i in range(0, len(available_gpus), args.tp_size):
            tmp = available_gpus[i:i + args.tp_size]
            gpu_id_lst.append(", ".join([str(gpu) for gpu in tmp]))
    
    return gpu_id_lst


def prepare_babilong_data(data_dir, tokenizer, inference_scaling=False, test_full=False):
    if not test_full:
        tasks = ['qa2', 'qa3', 'qa4', 'qa5', 'qa6', 'qa7']
        split_names = ['0k', '1k', '2k', '4k', '8k', '16k', '32k', '64k']
    else:
        tasks = ['qa1', 'qa2', 'qa3', 'qa4', 'qa5', 'qa6', 'qa7', 'qa8', 'qa9', 'qa10']
        split_names = ['0k', '1k', '2k', '4k', '8k', '16k', '32k', '64k', '128k']
    # tasks = ['qa2']
    # split_names = ['64k']
    all_input_texts = []

    for task in tqdm(tasks, desc='tasks'):
        # configure the prompt
        prompt_cfg = {
            'instruction': DEFAULT_PROMPTS[task]['instruction'],
            'examples': DEFAULT_PROMPTS[task]['examples'],
            'post_prompt': DEFAULT_PROMPTS[task]['post_prompt'],
            'template': LongMIT_DEFAULT_TEMPLATE,
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
                model_inputs = tokenizer(input_text, return_tensors="pt").input_ids
                # if inference_scaling:
                #     model_inputs = tokenizer.apply_chat_template(
                #         [{'role': 'system', 'content': BABILONG_SYSTEM_PROMPT},
                #         {'role': 'user', 'content': input_text}], 
                #         add_generation_prompt=True, tokenize=True, return_tensors="pt"
                #     )
                # else:
                #     model_inputs = tokenizer.apply_chat_template(
                #         [{'role': 'user', 'content': input_text}], 
                #         add_generation_prompt=True, tokenize=True, return_tensors="pt"
                #     )
                all_input_texts.append({"message": model_inputs, "golden": target, "task": task, "ctx_length": split_name, 'question': question})
    return all_input_texts             


def worker(gpu_ids: str, adapter_path: str, prompts_chunk, model_path, inference_args, return_list):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16, use_flash_attention_2="flash_attention_2", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    if adapter_path:
        model = PeftModelForCausalLM.from_pretrained(model, adapter_path)

    chunk_message = [item['message'] for item in prompts_chunk]

    logger.info(f"Start to generate {len(chunk_message)} samples")

    results = []
    for i in trange(len(chunk_message)):
        inp = chunk_message[i].to(model.device)
        res = model.generate(inp, **inference_args)
        pred_str = tokenizer.decode(res[0, inp.size(-1):], skip_special_tokens=True)
        prompts_chunk[i].pop("message")
        prompts_chunk[i]["pred"] = pred_str
        results.append(prompts_chunk[i])

    return_list.extend(results)


def main():
    parser = argparse.ArgumentParser(description="Inference with VLLM")
    parser.add_argument('--dataset_name', type=str, default=None, help='Name of the dataset')
    parser.add_argument('--model_path', type=str, default=None, help='Path to the model')
    parser.add_argument('--adapter_path', type=str, default=None, help='Path to the PEFT model')
    parser.add_argument('--save_path', type=str, default=None, help='Path to save the output')
    parser.add_argument('--tag', type=str, default=None, help='tag for the output file')
    parser.add_argument('--test_full', action='store_true', help='whether to test full dataset')
    parser.add_argument('--seed', type=int, default=27, help='Default seed value')
    parser.add_argument('--k', type=int, default=1, help='Number of generations per prompt')
    parser.add_argument('--num_gpus', type=int, default=8, help='Number of GPUs to use')
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default=None,
        help="Comma-separated list of GPU IDs (e.g., '0,1,2,7')",
    )
    parser.add_argument('--tp_size', type=int, default=1, help='Tensor parallel size')
    args = parser.parse_args()

    assert args.save_path is not None, "save_path is not set"
    
    auto_mkdir(args.save_path)

    torch.cuda.manual_seed_all(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    input_queries = prepare_babilong_data(args.dataset_name, tokenizer, test_full=args.test_full)
    if args.tag:
        out_file_path = os.path.join(args.save_path, f"preds_{os.path.basename(args.dataset_name)}_{args.tag}.jsonl")
    else:
        if args.adapter_path:
            suffix_tag = f"{args.adapter_path.split('/')[-2]}-{args.adapter_path.split('/')[-1]}"
            out_file_path = os.path.join(args.save_path, f"preds_{os.path.basename(args.dataset_name)}_{suffix_tag}_no_template.jsonl")
        else:
            suffix_tag = f"{args.model_path.split('/')[-2]}/{args.model_path.split('/')[-1]}"
            out_file_path = os.path.join(args.save_path, suffix_tag, "preds.jsonl")

    gpu_id_lst = [str(gpu_id) for gpu_id in args.gpu_ids.split(",")]
    num_gpus = len(gpu_id_lst)
    logger.info(gpu_id_lst)

    chunk_num = num_gpus // args.tp_size
    chunk_size = (len(input_queries) + chunk_num - 1) // chunk_num
    prompts_chunks = [input_queries[i*chunk_size:(i+1)*chunk_size] for i in range(chunk_num)]
  
    manager = mp.Manager()
    return_list = manager.list()
    processes = []

    # 使用 tqdm 显示总进度
    for chunk_id, gpu_ids in enumerate(gpu_id_lst):
        p = mp.Process(target=worker, args=(gpu_ids, args.adapter_path, prompts_chunks[chunk_id], args.model_path, inference_args['top_p'], return_list))
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
