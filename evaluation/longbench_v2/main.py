from modelzipper.tutils import *
# from utils import ALL_LB_TESTING_SETS, LB_DATA_PROMPT, LB_PRED_LEN, DATASET2CATEGORY
from utils import TEMPLATES
from datasets import load_dataset
import argparse
import torch
import torch.multiprocessing as mp
import numpy as np
from loguru import logger
from peft import PeftModelForCausalLM
import os


def chunks(lst, chunk_num):
    """Yield successive n-sized chunks from lst."""
    chunk_width = len(lst) // chunk_num
    ones = chunk_num - len(lst) % chunk_num 
    p = 0
    for i in range(chunk_num):
        if i == ones: chunk_width += 1
        yield lst[p: p + chunk_width]
        p += chunk_width


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def extract_answer(response):
    response = response.replace('*', '')
    match = re.search(r'The correct answer is \(([A-D])\)', response)
    if match:
        return match.group(1)
    else:
        match = re.search(r'The correct answer is ([A-D])', response)
        if match:
            return match.group(1)
        else:
            return None
        
def get_pred(rank=None, model_path=None, adapter_path=None, datasets=None, dataset_name=None, return_list=None, prompts_type = None):
    # os.environ["CUDA_VISIBLE_DEVICES"] = rank
    logger.info(f"gpu id {rank} is processing {dataset_name} length {len(datasets)} ...")
    # load models
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    logger.info(f"rank {rank} begin to load model ...")
    test_model = AutoModelForCausalLM.from_pretrained(model_path, use_flash_attention_2="flash_attention_2", device_map="auto").half().eval()
    if adapter_path:
        test_model = PeftModelForCausalLM.from_pretrained(test_model, adapter_path).eval()

    pred_res = []
    with torch.no_grad(), tqdm(total=len(datasets)) as pbar:
        for sample in datasets:
            if hasattr(test_model, "memory"):
                test_model.memory.reset()
            torch.cuda.empty_cache()
            context, answers = sample['context'],  sample['answer']

            prompt = TEMPLATES[prompts_type].replace('$DOC$', 
                                                     context.strip()).replace('$Q$', sample['question'].strip()).\
                                                        replace('$C_A$', sample['choice_A'].strip()).\
                                                            replace('$C_B$', sample['choice_B'].strip()).\
                                                                replace('$C_C$', sample['choice_C'].strip()).\
                                                                    replace('$C_D$', sample['choice_D'].strip())

            textual_input = tokenizer(prompt, return_tensors="pt").input_ids.to(test_model.device)

            outputs = test_model.generate(
                textual_input, 
                max_new_tokens=128, 
                temperature=0.9,
                do_sample = True,
                top_p = 0.95,
            )[0]

            pred_str = tokenizer.decode(outputs[textual_input.shape[-1]:], skip_special_tokens=True)
            pred = extract_answer(pred_str)
            pred_res.append({"dataset_name": dataset_name, 
                             "pred": pred, 
                             "output":pred_str,
                             "answers": answers, 
                             "judge": (pred == answers),
                            #  "context": context[:1000],
                             "length":sample["length"],
                             "difficulty":sample["difficulty"]}) 
            pbar.update(1)
            
    return_list.extend(pred_res)


#bash llama3.1-sft.sh


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="lb testing")
    parser.add_argument('--model_path', type=str, default=None, help='Path to the model')
    parser.add_argument('--adapter_path', type=str, default=None, help='Path to the PEFT model')
    parser.add_argument('--save_path', type=str, default=None, help='Path to save the output')
    parser.add_argument('--gpu_lst', type=str, default="0,1,2,3,4,5,6,7", help='All available gpus')
    parser.add_argument('--tp_size', type=int, default=2, help='model parallel size')
    parser.add_argument('--tag', type=str, default=None, help='output_dir tag')
    parser.add_argument('--chat_template', type=str, default=None, help='chat template')
    parser.add_argument('--model_max_length_setting', type=str, default="normal_setting", help='Model max length setting')
    parser.add_argument('--seed', type=int, default=27, help='default seed')
    parser.add_argument("--cot", "-cot", action='store_true') # set to True if using COT
    parser.add_argument("--no_context", "-nc", action='store_true') # set to True if using no context (directly measuring memorization)
    parser.add_argument("--rag", "-rag", type=int, default=0) # set to 0 if RAG is not used, otherwise set to N when using top-N retrieved context
    
    args = parser.parse_args()
    args = parser.parse_args()
    print("Pid:", os.getpid())
    mp.set_start_method('spawn', force=True)

    all_gpu_list = args.gpu_lst.split(',')
    logger.info(f'begin to eval on {len(all_gpu_list)} gpus | tensor parallel size is {args.tp_size}...')
    split_gpu_list = []
    for i in range(0, len(all_gpu_list), args.tp_size):
        split_gpu_list.append(",".join(all_gpu_list[i:i + args.tp_size]))
    print("split_gpu_list:", split_gpu_list)
    world_size = len(split_gpu_list)

    if args.tag:
        pred_dir = os.path.join(args.save_path, args.tag)
    else:
        if args.adapter_path:
            suffix_tag = f"{args.adapter_path.split('/')[-2]}-{args.adapter_path.split('/')[-1]}"
            pred_dir = os.path.join(args.save_path, suffix_tag)
        else:
            suffix_tag = f"{args.model_path.split('/')[-2]}-{args.model_path.split('/')[-1]}"
            pred_dir = os.path.join(args.save_path, suffix_tag)

    seed_everything(args.seed)
    os.makedirs(args.save_path, exist_ok=True)
    
    if args.rag > 0:
        prompts_type = "rag"
    elif args.no_context:
        prompts_type = "no_context"
    elif args.cot:
        prompts_type = "0shot_cot"
    else:
        prompts_type = "0shot"

    # dataset = load_dataset('/data/pub_data/LongBench-v2', split='train')
    dataset = load_dataset('THUDM/LongBench-v2', split='train')
    data_all = dataset.filter(lambda x: (x['length'] != 'long') and (len(x['context']) < 128000 * 4))

    tokenzier = AutoTokenizer.from_pretrained(args.model_path)
    new_data = []
    for sample in tqdm(data_all):
        if tokenzier(sample['context'], return_tensors='pt').input_ids.shape[1]<=128000:
            new_data.append(sample)

    test_datasets = ['Code Repository Understanding', 
                     'Long In-context Learning',
                     'Long Structured Data Understanding',
                     'Long-dialogue History Understanding',
                     'Multi-Document QA',
                     'Single-Document QA']

    for dataset_name in test_datasets:
        test_data = [k for k in new_data if k['domain'] == dataset_name]
        save_res_path = os.path.join(pred_dir, dataset_name + ".jsonl")

        data_subsets = list(chunks(test_data, world_size))

        with tqdm(total=world_size) as pbar:
            processes = []
            manager = mp.Manager()
            return_list = manager.list()
             
            for rank in range(0, world_size):
                os.environ["CUDA_VISIBLE_DEVICES"] = split_gpu_list[rank]
                p = mp.Process(target=get_pred, args=(split_gpu_list[rank], args.model_path, args.adapter_path, data_subsets[rank], dataset_name, return_list, prompts_type))
                p.start()
                processes.append(p)
                time.sleep(5)
            
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