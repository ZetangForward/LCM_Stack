import json
from dataclasses import dataclass, field
from typing import List, Optional
from datasets import load_dataset, load_from_disk, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, LlamaTokenizer
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests, os
from modelzipper.tutils import *
import sys
from chat import apply_chat_template

tqdm.pandas()

@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    url: Optional[str] = field(
        default="http://localhost",
        metadata={"help": "url of the model response"},
    )
    tokenizer: Optional[str] = field(
        default="HuggingFaceH4/mistral-7b-sft-beta",
        metadata={"help": "the tokenizer to use"},
    )
    ports: List[str] = field(default_factory=lambda: ["8000"], metadata={"help": "ports of the model response"})
    eos_ids: List[int] = field(default_factory=lambda: [], metadata={"help": "the ids of the end of sentence tokens"})
    input_data_dir: Optional[str] = field(
        default="cornfieldrm/iterative-prompt-v1-iter1-2K",
        metadata={"help": "the location of the dataset name or path"},
    )
    output_dir: Optional[str] = field(
        default="uf_split0_responses_K8.json",
        metadata={"help": "the location of the output file"},
    )
    bos_format: Optional[str] = field(
        default="",
        metadata={"help": "the format of the beginning of the sentence"},
    )
    K: Optional[int] = field(
        default=8,
        metadata={"help": "the number of generations per prompt"},
    )
    max_input_length: Optional[int] = field(
        default=10000,
        metadata={"help": "the maximum length of the input tokens"},
    )
    max_new_tokens: Optional[int] = field(
        default=512,
        metadata={"help": "the maximum length of the new tokens"},
    )
    min_new_tokens: Optional[int] = field(
        default=16,
        metadata={"help": "the minmum length of the new tokens"},
    )
    seed: Optional[int] = field(
        default=42,
        metadata={"help": "the random seed"},
    )
    temperature: Optional[float] = field(
        default=0.7,
        metadata={"help": "the temperature"},
    )
    use_beam_search: Optional[bool] = field(
        default=False,
        metadata={"help": "the beam search"},
    )
    model_type: Optional[str] = field(
        default="llama-2",
        metadata={"help": "model type"},
    )
    dataset_key: Optional[str] = field(
        default="context_messages",
        metadata={"help": "the key of the dataset"},
    )
    max_workers: Optional[int] = field(
        default=64,
        metadata={"help": "the number of workers"},
    )
    strategy: Optional[str] = field(
        default='full',
        metadata={"help": 'the strategy to generate the data'},
    )


def create_path(directories):
    return  os.path.join(*directories)

def query_model(item, args, port):
    json = {
        **args,
        "prompt": item["prompt"],
    }
    response = requests.post(url=script_args.url + ":" + str(port) + "/generate", json=json)
    response_json = response.json()
    pred = response_json["text"][0][len(item["prompt"]) :]
    # post process
    if '<|eot_id|>' in pred:
        pred = pred.split('<|eot_id|>')[0]
    
    return dict(
        question = item['question'],
        predict = pred,
        label = item['label'],
        context_lst = item['context_lst'],
        meta_info = item["meta_info"],
    )

def merge_chat_item(item, model_type, tokenizer, concate_str=' \n ', meta_info=None, num_save_chunks=12):
    question, answer, context_lst = item["question"], item["answer"], item["context"]
    all_merged_res = []
    for i in range(len(context_lst)):
        q, a, c_lst = question[i], answer[i], context_lst[i]
        if num_save_chunks < 0:
            c_lst = sorted(c_lst[num_save_chunks:], key=lambda x: x['chunk_id'])
        else:
            c_lst = sorted(c_lst[:num_save_chunks], key=lambda x: x['chunk_id'])
        c_lst = [chunk['chunk'] for chunk in c_lst]
        context = concate_str.join(c_lst)
        prompt = [{'role': 'user', 'content': f'Answer the question according to the context below:\n{context}\n Question: {q}'}]
        # chat_prompt = tokenizer.apply_chat_template(conversation=prompt, tokenize=False, 
        # add_generation_prompt=True)
        chat_prompt = apply_chat_template(
            model_type, 
            messages=prompt,
            tokenizer=tokenizer,
            add_generation_prompt=True,
        ).raw
        # fastchat template
        all_merged_res.append({'prompt': chat_prompt, 'context_lst': c_lst, 'question': q, 'label': a, 'meta_info': meta_info})
    return all_merged_res


if __name__ == '__main__':
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    ports = script_args.ports

    tokenizer = AutoTokenizer.from_pretrained(script_args.tokenizer)
    default_args = {
        "use_beam_search": script_args.use_beam_search,
        "n": script_args.K,
        "temperature": script_args.temperature,
        "max_tokens": script_args.max_new_tokens,
        "min_tokens": script_args.min_new_tokens,
        "seed": script_args.seed,
        "top_p": 0.95,
        "top_k": -1,
        "stop_token_ids": [tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
    }
    
    # load datasets
    print('Loading datasets ...')
    critical_data_names = ['gpt-multi_detail_paper_short', 'gpt-bio_book', 'gpt-multi_detail_paper_long', 'longalpaca-train', 'gpt-multi_detail_book', 'gpt-one_detail_paper', 'gpt-one_detail_book']
    with tqdm(total=len(critical_data_names), desc='Loading datasets') as pbar:
        for dataset_name in critical_data_names:
            dataset_path = f'{script_args.input_data_dir}/{dataset_name}'
            if os.path.isdir(dataset_path):
                content = load_from_disk(dataset_path)
            else:
                content = auto_read_data(dataset_path)
            
            if script_args.strategy == 'full':  # generate with full critical paths
                num_save_chunks = 6
                output_file_path = os.path.join(script_args.output_dir, 'pred_w_full_paths', dataset_name)
            elif script_args.strategy == 'half':  # generate with half critical paths
                num_save_chunks = 3
                output_file_path = os.path.join(script_args.output_dir, 'pred_w_half_paths', dataset_name)
            elif script_args.strategy == 'wrong':  # generate with no critical paths
                print('generate with no critical paths')
                num_save_chunks = -3
                output_file_path = os.path.join(script_args.output_dir, 'pred_wo_critical_paths', dataset_name)

            ds, gathered_data = [], []
            for item in tqdm(content, desc=f'Processing {dataset_name}'):
                ds.extend(
                    merge_chat_item(
                        item, script_args.model_type, tokenizer=tokenizer, num_save_chunks=num_save_chunks, 
                        meta_info={'dataset_name': dataset_name, 'process_turns': 1, 'path_nums': num_save_chunks}
                    )
                )  

            with ThreadPoolExecutor(max_workers=script_args.max_workers) as executor:
                result = [executor.submit(query_model, ds[i], default_args, ports[i % len(ports)]) for i in range(len(ds))]
                for _ in tqdm(as_completed(result), total=len(result)): pass  # use tqdm to show progress
                responses = [r.result() for r in result]

            for i in range(len(ds)):
                gathered_data.append(responses[i])
            
            print('Have collected ', len(gathered_data), 'samples, begin to save ...')
            data_dict = {key: [dic[key] for dic in gathered_data] for key in gathered_data[0]}
            Dataset.from_dict(data_dict).save_to_disk(output_file_path)
            pbar.update(1)


    
    

    
    
    
    
    