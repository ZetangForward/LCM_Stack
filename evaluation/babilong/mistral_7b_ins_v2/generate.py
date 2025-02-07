from modelzipper.tutils import *
from fire import Fire
import sys
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
import requests, os 
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from multiprocessing import Pool
from multiprocessing import get_context
import copy

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def query_model(item, args, client):
    try:
        json = {
            "model": 'llama',
            **args,
            "messages": item.pop('input_text'),
        }
        response = client.chat.completions.create(**json)    
        pred = response.choices[0].message.content
        item['pred'] = pred
    except Exception as e:
        item['pred'] = ""
    
    return item


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="vllm inference")
    parser.add_argument('--ports', type=int, nargs='+', default=[4100, 4101], help='Ports for the model')
    parser.add_argument('--url', type=str, default="http://localhost", help='url path')
    parser.add_argument('--dataset_name', type=str, default=None, help='Dataset name')
    parser.add_argument('--data_dir', type=str, default=None, help='Path to the PEFT model')
    parser.add_argument('--task_name', type=str, default=None, help='task name')
    parser.add_argument('--model_path', type=str, default=None, help='Path to the model')
    parser.add_argument('--peft_path', type=str, default=None, help='Path to the PEFT model')
    parser.add_argument('--save_path', type=str, default=None, help='Path to save the output')
    parser.add_argument('--rope_theta', type=float, default=None, help='RoPE theta value')
    parser.add_argument('--rope_factor', type=float, default=None, help='RoPE factor')
    parser.add_argument('--rope_type', type=str, default=None, help='RoPE type')
    parser.add_argument('--max_position_embeddings', type=int, default=None, help='Maximum position embeddings')
    parser.add_argument('--model_max_length_setting', type=str, default="normal_setting", help='Model max length setting')
    parser.add_argument('--max_training_length', type=int, default=8192, help='Maximum training length')
    parser.add_argument('--seed', type=int, default=27, help='default seed')
    parser.add_argument('--max_workers', type=int, default=2, help='max number of workers')
    parser.add_argument('--use_logn', action='store_true', help='use logn')

    args = parser.parse_args()
    if args.max_position_embeddings == -1:
        args.max_position_embeddings = None
    if args.rope_theta == -1:
        args.rope_theta = None
    
    world_size = torch.cuda.device_count()
    mp.set_start_method('spawn', force=True)
    
    print(f'begin to eval on {world_size} gpus ...')
   
    assert args.save_path is not None, "save_path is not set"
    
    auto_mkdir(args.save_path)

    folder_name = os.path.join(args.data_dir, args.dataset_name, args.task_name)

    all_content = dict()
    for file_name in os.listdir(folder_name):
        all_content[file_name.split('.')[0]] = auto_read_data(os.path.join(folder_name, file_name))

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    ports = args.ports
    torch.cuda.manual_seed_all(args.seed)

    url_ports = [args.url + ":" + str(port) + "/v1" for port in ports]
    clients = [OpenAI(api_key="EMPTY", base_url=url_port) for url_port in url_ports]

    out_file_path = os.path.join(args.save_path, f"preds_{args.dataset_name}.jsonl")
    input_queries, results = [], []
    for testing_setting, dataset_content in all_content.items():
        for bucket_id, bucket_sample in dataset_content.items():
            instruction = bucket_sample['system_prompt']
            quries = bucket_sample['query']
            call_parameters = bucket_sample['call_parameters']
            source_docs = bucket_sample['source_docs']
            for id_, q in enumerate(quries):
                if isinstance(instruction, list):
                    messages = copy.deepcopy(instruction)
                    messages.append({'role': 'user', 'content': q})
                else:
                    messages=[
                        {'role': 'system', 'content': instruction}, 
                        {'role': 'user', 'content': q}
                    ]
                input_queries.append({'bucket_id': bucket_id, 'query': q, "input_text": messages, 'testing_setting': testing_setting, 'dataset_name': args.dataset_name, 'call_parameters': call_parameters[id_], 'source_docs': source_docs[id_]})

    logger.info(f"Total number of queries: {len(input_queries)}")
    
    default_args = {
        "n": 1,
        "temperature": 0.7,
        "max_tokens": 2000,
        "seed": 42,
        "top_p": 0.95,
    }

    print('Begin to query the model ...')

    # one_sample = query_model(input_queries[0], default_args, clients[0])
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        result = [executor.submit(query_model, input_queries[i], default_args, clients[i % len(ports)]) for i in range(len(input_queries))]
        for _ in tqdm(as_completed(result), total=len(result)): pass  # use tqdm to show progress
        gathered_data = [r.result() for r in result]

    print('Have collected ', len(gathered_data), 'samples, begin to save ...')
    
    auto_save_data(gathered_data, out_file_path)