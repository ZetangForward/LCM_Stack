import argparse
import datasets
import sys
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from tqdm import tqdm
from longppl_utils import *
import os
import multiprocessing as mp
from peft import PeftModelForCausalLM
def chunks(lst, chunk_num):
    """Yield successive n-sized chunks from lst."""
    chunk_width = len(lst) // chunk_num
    ones = chunk_num - len(lst) % chunk_num 
    p = 0
    for i in range(chunk_num):
        if i == ones: chunk_width += 1
        yield lst[p: p + chunk_width]
        p += chunk_width


def compute_perplexity(
    encodings, model, evaluator_model, tokenizer, evaluator_tokenizer, args, device=None
):
    if device is not None:
        assert device in ["gpu", "cpu",
                          "cuda"], "device should be either gpu or cpu."
        if device == "gpu":
            device = "cuda"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    encoded_texts = [x[0:args.max_length-1] for x in encodings["input_ids"]]

    pbar = tqdm(total=len(encoded_texts))
    longppls, ppls, nums_key_token, nums_token = [], [], [], []

    def convert_tokenized_to_text(tokenized_input, tokenizer_path):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        text = tokenizer.batch_decode(tokenized_input)
        return text

    for encoding_index in range(0, len(encoded_texts)):
        tokenized_input = torch.tensor(encoded_texts[encoding_index:encoding_index+1]).to(device)
        if args.tokenized:
            text = convert_tokenized_to_text(tokenized_input, args.llama_path)
        else:
            text = convert_tokenized_to_text(tokenized_input, args.model)

        if not os.path.exists(os.path.join(args.key_text_path, args.evaluator_name)):
            os.makedirs(os.path.join(args.key_text_path, args.evaluator_name))
            
        save_path = os.path.join(args.key_text_path, args.evaluator_name, f"slice_{encoding_index}.txt")

        with torch.no_grad():
            output = compute_longppl(
                text=text[0], 
                model=model,
                evaluator_model=evaluator_model,
                tokenizer=tokenizer, 
                evaluator_tokenizer=evaluator_tokenizer, 
                save_path=save_path, 
                trunc_len=args.trunc_len, 
                sliding_window=args.sliding_window
            )
        longppl = output['longppl']
        ppl = output['ppl']
        n_key_token = output['n_key_token'] 
        n_token = output['n_token']
        
        if longppl is not None:
            longppls.append(longppl)
            nums_key_token.append(n_key_token)
        ppls.append(ppl)
        nums_token.append(n_token)
        longppl = (np.stack(longppls) * np.stack(nums_key_token)).sum() / np.stack(nums_key_token).sum()
        ppl = (np.stack(ppls) * np.stack(nums_token)).sum() / np.stack(nums_token).sum()

        pbar.set_postfix(longppl=longppl, ppl=ppl)
        pbar.update(1)

    return {"longppls":longppls,
            "nums_key_token":nums_key_token,
            "ppls": ppls,
            "nums_token":nums_token}

    return {"longppl": longppl, "ppl": ppl}



def worker(args, input_texts, result_list, model_config, adapter_path,use_yarn):
    # model = AutoModelForCausalLM.from_pretrained(
    #     args.model,
    #     torch_dtype=torch.bfloat16,
    #     device_map="auto",
    #     trust_remote_code=True,
    #     attn_implementation="flash_attention_2"
    # )
    config=model_config if (model_config is not None and len(model_config)>0) else None
    if use_yarn:
        pass
        # config = "/mnt/petrelfs/tangzecheng/LCM_Stack/evaluation/LongPPL/yarn_config.json"
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, use_flash_attention_2="flash_attention_2", device_map="auto",  
    config = AutoConfig.from_pretrained(config) if config else None)

    if adapter_path:
        model = PeftModelForCausalLM.from_pretrained(model, adapter_path).eval()
    print("model_config:", config)
        
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    if args.mode == 'online':
        evaluator_model = AutoModelForCausalLM.from_pretrained(args.evaluator_model, torch_dtype=torch.bfloat16, device_map="auto")
    elif args.mode == 'offline':
        evaluator_model = None
    evaluator_tokenizer = AutoTokenizer.from_pretrained(args.evaluator_model)
    

    ppls = compute_perplexity(
        model=model, 
        evaluator_model=evaluator_model, 
        tokenizer=tokenizer, 
        evaluator_tokenizer=evaluator_tokenizer,
        encodings=input_texts,
        args=args,
    )
    result_list.append(ppls)
    # print(f"{args.model}: longppl: {ppl['longppl']}, ppl: {ppl['ppl']}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--gpu-lst",type=str)
    parser.add_argument("--adapter-path",type=str, default = "")
    parser.add_argument("--tp-size",type=int, default=1)
    parser.add_argument("--evaluator-model", type=str, default = "Crystalcareai/meta-llama-3.1-8b")
    parser.add_argument("--evaluator-name", type=str, help='To use the offline key tokens we provided, set it to Qwen2-72B-Instruct, Mistral-Large-Instruct-2407, or Meta-Llama-3.1-8B', default="Meta-Llama-3.1-8B")
    parser.add_argument("--mode", type=str, choices=['online', 'offline'], default='offline')
    parser.add_argument("-d", "--dataset", type=str)
    parser.add_argument("-s", "--subset", type=str)
    parser.add_argument("-f", "--feature", type=str)
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--dataset-min-tokens", type=int)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--samples", type=int)
    parser.add_argument("--save-tokenized", type=str)
    parser.add_argument("--tokenized", type=str, default = "")
    parser.add_argument("--trunc-len", type=int, default=4096)
    parser.add_argument("--sliding-window", type=int, default=1024)
    parser.add_argument("--llama-path", type=str, default="meta-llama/Llama-2-7b-hf")
    
    args = parser.parse_args()
    print("Adapter Path: ", args.adapter_path)

    args.key_text_path = "/mnt/petrelfs/tangzecheng/LCM_Stack/evaluation/LongPPL/LongPPL/perplexity/key_text"
    print("Pid:", os.getpid())
    mp.set_start_method('spawn', force=True)

    all_gpu_list = args.gpu_lst.split(',')
    print(f'begin to eval on {len(all_gpu_list)} gpus | tensor parallel size is {args.tp_size}...')
    split_gpu_list = []
    for i in range(0, len(all_gpu_list), args.tp_size):
        split_gpu_list.append(",".join(all_gpu_list[i:i + args.tp_size]))
    print("split_gpu_list:", split_gpu_list)
    world_size = len(split_gpu_list)

    input_texts = datasets.load_dataset(args.tokenized)

    if args.dataset_min_tokens:
        input_texts = input_texts.filter(
            lambda x: x["tokenized_len"] >= args.dataset_min_tokens)

    if args.samples:
        input_texts = input_texts['test'][:args.samples]

    input_texts = [{k:v[i] for k,v in input_texts.items()} for i in range(args.samples)]

    input_texts = list(chunks(input_texts, world_size))
    
    chunk_texts = []

    for chunk_input_texts in input_texts:
        temp ={k:[] for k in chunk_input_texts[0].keys()}
        for sample in chunk_input_texts:
            for k,v in sample.items():
                temp[k].append(v)
        chunk_texts.append(temp)
    
    use_yarn = False

    if args.adapter_path:
        model_config = os.path.join(args.adapter_path, "config.json")
    else:
        model_config = None

    manager = mp.Manager()
    result_list = manager.list()
    processes = []
    for  gpus, chunk_text in zip(split_gpu_list, chunk_texts):
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
        process = mp.Process(
            target = worker,
            args = (args, chunk_text, result_list, model_config, args.adapter_path,use_yarn)
        )
        process.start()
        processes.append(process)
    
    for p in processes:p.join()

    
    from collections import defaultdict

    results = defaultdict(list)
    for dic in result_list:
        for k,v in dic.items():
            results[k].extend(v)
    

    longppl = (np.stack(results['longppls']) * np.stack(results['nums_key_token'])).sum() / np.stack(results['nums_key_token']).sum()
    ppl = (np.stack(results['ppls']) * np.stack(results['nums_token'])).sum() / np.stack(results['nums_token']).sum()

    import pandas as pd

    if args.adapter_path:
        args.model = args.adapter_path
    save_name = "./results/" +"-".join(args.model.strip("/").split("/")[-2:])
    save_dir = os.makedirs(save_name, exist_ok= True)
    pd.DataFrame({
        "longppl":[f"{longppl:.2f}"],
        "ppl":[f"{ppl:.2f}"]
    }).to_csv(save_name + '/ppls.csv')

    print("保存成功:", save_name + '/ppls.csv')

