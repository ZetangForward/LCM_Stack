import argparse
import json
import multiprocessing as mp
import os
import copy
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from modelzipper.tutils import *
from loguru import logger

inference_args = dict(
    top_p = dict(
        n = 1, 
        temperature = 0.7, 
        max_tokens = 512, 
        seed = 42, 
        top_p = 0.95,
    ),
    greedy = dict(
         n = 1,
        temperature = 0.0,
        max_tokens = 512,
        seed = 42,
    )
)


def worker(gpu_ids, prompts_chunk, model_path, model_args, inference_args, return_list):
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
        results.append(original_prompt_data)

    return_list.extend(results)

def main():
    parser = argparse.ArgumentParser(description="使用 vllm 对 prompts 进行生成（数据并行）")
    parser.add_argument('--dataset_name', type=str, default=None, help='Dataset name')
    parser.add_argument('--data_dir', type=str, default=None, help='Path to the PEFT model')
    parser.add_argument('--task_name', type=str, default=None, help='task name')
    parser.add_argument('--model_path', type=str, default=None, help='Path to the model')
    parser.add_argument('--peft_path', type=str, default=None, help='Path to the PEFT model')
    parser.add_argument('--save_path', type=str, default=None, help='Path to save the output')
    parser.add_argument('--seed', type=int, default=27, help='default seed')
    parser.add_argument('--max_workers', type=int, default=2, help='max number of workers')
    parser.add_argument('--use_logn', action='store_true', help='use logn')
    parser.add_argument('--temperature', type=float, default=1.0, help='生成的温度参数')
    parser.add_argument('--k', type=int, default=1, help='每个 prompt 生成的数量 K')
    parser.add_argument('--num_gpus', type=int, default=8, help='使用的 GPU 数量')
    parser.add_argument('--tp_size', type=int, default=1, help='tensor_parallel_size')
    args = parser.parse_args()

    assert args.save_path is not None, "save_path is not set"
    
    auto_mkdir(args.save_path)

    folder_name = os.path.join(args.data_dir, args.dataset_name, args.task_name)

    all_content = dict()
    for file_name in os.listdir(folder_name):
        all_content[file_name.split('.')[0]] = auto_read_data(os.path.join(folder_name, file_name))

    torch.cuda.manual_seed_all(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    out_file_path = os.path.join(args.save_path, f"preds_{args.dataset_name}.jsonl")
    input_queries = []
    for testing_setting, dataset_content in all_content.items():
        for bucket_id, bucket_sample in dataset_content.items():
            instruction = bucket_sample['system_prompt']
            quries = bucket_sample['query']
            call_parameters = bucket_sample['call_parameters']
            source_docs = bucket_sample['source_docs']
            for id_, q in enumerate(quries):
                if isinstance(instruction, list):
                    message = copy.deepcopy(instruction)
                    message.append({'role': 'user', 'content': q})
                else:
                    message=[
                        {'role': 'system', 'content': instruction}, 
                        {'role': 'user', 'content': q}
                    ]
                tokenized_message = tokenizer.apply_chat_template(message, add_generation_prompt=True, tokenize=False)
                input_queries.append({'bucket_id': bucket_id, 'query': q, "message": tokenized_message, 'testing_setting': testing_setting, 'dataset_name': args.dataset_name, 'call_parameters': call_parameters[id_], 'source_docs': source_docs[id_]})

    logger.info(f"Total number of queries: {len(input_queries)}")

    model_args = {
        "tensor_parallel_size": args.tp_size, 
        "gpu_memory_utilization": 0.98,
        "max_model_len": 128000, 
        "trust_remote_code": True, 
    }

    num_gpus = args.num_gpus
    chunk_size = (len(input_queries) + num_gpus - 1) // num_gpus
    prompts_chunks = [input_queries[i*chunk_size:(i+1)*chunk_size] for i in range(num_gpus)]
  
    manager = mp.Manager()
    return_list = manager.list()
    processes = []

    # construct gpu_ids list
    if args.tp_size == 1:
        gpu_id_lst = [str(i) for i in range(num_gpus)]
    else:
        gpu_id_lst = []

        for i in range(0, num_gpus, args.tp_size):
            tmp = list(range(i, i + args.tp_size))
            gpu_id_lst.append(", ".join([str(i) for i in tmp]))

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
