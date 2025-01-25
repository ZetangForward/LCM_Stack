from modelzipper.tutils import *
# from utils import ALL_LB_TESTING_SETS, LB_DATA_PROMPT, LB_PRED_LEN, DATASET2CATEGORY
from utils import TEMPLATES, PRED_LENGTH
from datasets import load_dataset
import argparse
import torch
import torch.multiprocessing as mp
import numpy as np
from loguru import logger
from peft import PeftModelForCausalLM
import os

context_max_length = {"8k_setting": 7200, "tiny_setting": 15500, "normal_setting": 32000, "long_setting": 63500, "ultra_long_setting": 127500}
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
        
def get_pred(rank=None, model_path=None, adapter_path=None, datasets=None, dataset_name=None, max_context_length=None, chat_template=None, return_list=None, prompts_type = None):
    os.environ["CUDA_VISIBLE_DEVICES"] = rank
    logger.info(f"gpu id {rank} is processing {dataset_name} length {len(datasets)} ...")
    # load models
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    logger.info(f"rank {rank} 开始加载模型 ...")
    test_model = AutoModelForCausalLM.from_pretrained(model_path, use_flash_attention_2="flash_attention_2", device_map="auto").half().eval()
    if adapter_path:
        test_model = PeftModelForCausalLM.from_pretrained(test_model, adapter_path).eval()

    if hasattr(test_model, "generation_config"):
        eos_token_id = test_model.generation_config.eos_token_id
    else:
        eos_token_id = tokenizer.eos_token_id

    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
        
    eos_token_id.append(tokenizer.encode("\n", add_special_tokens=False)[-1])
    # PROMPT_TEMPLATE, PRED_LENGTH = LB_DATA_PROMPT[dataset_name], LB_PRED_LEN[dataset_name]
    
    pred_res = []
    with torch.no_grad(), tqdm(total=len(datasets)) as pbar:
        for sample in datasets:
            if hasattr(test_model, "memory"):
                test_model.memory.reset()
            torch.cuda.empty_cache()
            context, answers = sample['context'],  sample['answer']
            if  "length" in sample:
                length = sample["length"]
            else:
                length = 0
            # prompt = PROMPT_TEMPLATE.format(input=input_, context=context)
            prompt = TEMPLATES[prompts_type].replace('$DOC$', 
                                                     context.strip()).replace('$Q$', sample['question'].strip()).\
                                                        replace('$C_A$', sample['choice_A'].strip()).\
                                                            replace('$C_B$', sample['choice_B'].strip()).\
                                                                replace('$C_C$', sample['choice_C'].strip()).\
                                                                    replace('$C_D$', sample['choice_D'].strip())
        
            # if (not DATASET2CATEGORY[dataset_name] in ["EN Few-Shot Learning", "Code Completion"]):
                # if tokenizer.chat_template is not None:
                #     prompt = tokenizer.apply_chat_template(
                #         [{'role': 'user', 'content': prompt}],
                #         add_generation_prompt=True, tokenize=False
                #     )
                # elif chat_template is not None:
                #     prompt = chat_template.format(prompt)

            textual_input = tokenizer(prompt, return_tensors="pt").input_ids[0].to(test_model.device)

            if len(textual_input) > max_context_length - PRED_LENGTH - 100:
                # continue
                half = int((max_context_length - PRED_LENGTH - 100)/2)
                prompt = tokenizer.decode(textual_input[:half], skip_special_tokens=True) + tokenizer.decode(textual_input[-half:], skip_special_tokens=True)

            input_ids = tokenizer(prompt, return_tensors="pt").to(test_model.device).input_ids
            if input_ids.size(-1) == 0:
                print("=============")
                print(f"textual_input.shape {textual_input.shape}")
                print(f"max_context_length {max_context_length}")
                print("=============")
            print(f"context length: {input_ids.shape}")

            # if dataset_name in ["2wikimqa_e", "hotpotqa_e", "musique_e", "multifieldqa_en_e", "qasper_e", "narrativeqa_e", "samsum_e"]:
            if True:
                outputs = test_model.generate(
                    input_ids, 
                    max_new_tokens=PRED_LENGTH, 
                    do_sample=None,
                    begin_suppress_tokens=eos_token_id,
                    eos_token_id=eos_token_id, 
                    temperature=0.1,
                    top_p=None,
                )[0]
            else:
                outputs = test_model.generate(
                    input_ids,
                    max_new_tokens=PRED_LENGTH,
                    eos_token_id=eos_token_id,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                )[0]

            pred_str = tokenizer.decode(outputs[input_ids.shape[-1]:], skip_special_tokens=True)
            pred = extract_answer(pred_str)
            pred_res.append({"dataset_name": dataset_name, 
                             "pred": pred, 
                             "answers": answers, 
                             "judge": (pred == answers),
                             "context": context[:1000],
                             "length":sample["length"],
                             "difficulty":sample["difficulty"],
                             "index":sample["index"]}) 
            pbar.update(1)
            
    return_list.extend(pred_res)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="lb testing")
    parser.add_argument('--model_path', type=str, default=None, help='Path to the model')
    parser.add_argument('--adapter_path', type=str, default=None, help='Path to the PEFT model')
    parser.add_argument('--save_path', type=str, default=None, help='Path to save the output')
    parser.add_argument('--gpu_lst', type=str, default="0,1,2,3,4,5,6,7", help='All available gpus')
    parser.add_argument('--tp_size', type=int, default=1, help='model parallel size')
    parser.add_argument('--tag', type=str, default=None, help='output_dir tag')
    parser.add_argument('--chat_template', type=str, default=None, help='chat template')
    parser.add_argument('--model_max_length_setting', type=str, default="normal_setting", help='Model max length setting')
    parser.add_argument('--seed', type=int, default=27, help='default seed')
    parser.add_argument("--cot", "-cot", action='store_true') # set to True if using COT
    parser.add_argument("--no_context", "-nc", action='store_true') # set to True if using no context (directly measuring memorization)
    parser.add_argument("--rag", "-rag", type=int, default=0) # set to 0 if RAG is not used, otherwise set to N when using top-N retrieved context

    args = parser.parse_args()
    args = parser.parse_args()
    
    mp.set_start_method('spawn', force=True)

    all_gpu_list = args.gpu_lst.split(',')
    logger.info(f'begin to eval on {len(all_gpu_list)} gpus | tensor parallel size is {args.tp_size}...')
    split_gpu_list = []
    for i in range(0, len(all_gpu_list), args.tp_size):
        split_gpu_list.append(",".join(all_gpu_list[i:i+args.tp_size]))

    world_size = len(split_gpu_list)

    if args.tag:
        pred_dir = os.path.join(args.save_path, args.tag)
    else:
        if args.adapter_path:
            suffix_tag = f"{args.adapter_path.split('/')[-2]}-{args.adapter_path.split('/')[-1]}"
            pred_dir = os.path.join(args.save_path, suffix_tag)
        else:
            pred_dir = os.path.join(args.save_path, "vanilla")
    
    # if os.path.exists(pred_dir):
    #     already_finish_files = auto_read_dir(pred_dir, file_suffix=".jsonl")
    #     already_finish_files = [os.path.basename(f).split('.')[0] for f in already_finish_files]
        
    #     # check generated cases
    #     for f in already_finish_files[::-1]:
    #         num_test_cases = len(load_dataset('THUDM/LongBench', f, split='test'))
    #         num_pred_cases = len(auto_read_data(os.path.join(pred_dir, f + ".jsonl")))
    #         if num_test_cases != num_pred_cases: 
    #             print(f"{f} has not been processed, removing it from finished files ...")
    #             already_finish_files.remove(f)
    # else:
    #     auto_mkdir(pred_dir)
    #     already_finish_files = []
    
    # test_datasets = list(set(ALL_LB_TESTING_SETS) - set(already_finish_files))
    # logger.info(f"evaluating on datasets: {test_datasets}")
    # max_context_length = context_max_length[args.model_max_length_setting]
    # logger.info(f"max_context_length are set as {max_context_length}")
    


    seed_everything(args.seed)



    os.makedirs(args.save_path, exist_ok=True)
    print(args)
    if args.rag > 0:
        # out_file = os.path.join(args.save_path, args.model_path.split("/")[-1] + f"_rag_{str(args.rag)}.jsonl")
        prompts_type = "rag"
    elif args.no_context:
        # out_file = os.path.join(args.save_path, args.model_path.split("/")[-1] + "_no_context.jsonl")
        prompts_type = "no_context"
    elif args.cot:
        # NOTE: prompts 还没改
        # out_file = os.path.join(args.save_path, args.model_path.split("/")[-1] + "_cot.jsonl")
        prompts_type = "0shot_cot"
    else:
        # out_file = os.path.join(args.save_path, args.model_path.split("/")[-1] + ".jsonl")
        prompts_type = "0shot"


    dataset = json.load(open('/data/pub_data/LongBench-v2/data.json', 'r', encoding='utf-8'))
    data_all = [{"_id": item["_id"], "domain": item["domain"], "sub_domain": item["sub_domain"], "difficulty": item["difficulty"], "length": item["length"], "question": item["question"], "choice_A": item["choice_A"], "choice_B": item["choice_B"], "choice_C": item["choice_C"], "choice_D": item["choice_D"], "answer": item["answer"], "context": item["context"],"index": i} for i, item in enumerate(dataset)]


    test_datasets = ['Code Repository Understanding', 
                     'Long In-context Learning',
                     'Long Structured Data Understanding',
                     'Long-dialogue History Understanding',
                     'Multi-Document QA',
                     'Single-Document QA']

    max_context_length = context_max_length[args.model_max_length_setting]

    for dataset_name in test_datasets:
        test_data = [k for k in data_all if k['domain'] == dataset_name]
        save_res_path = os.path.join(pred_dir, dataset_name + ".jsonl")

        data_subsets = list(chunks(test_data, world_size))

        with tqdm(total=world_size) as pbar:
            processes = []
            manager = mp.Manager()
            return_list = manager.list()
             
            for rank in range(0, world_size):
                p = mp.Process(target=get_pred, args=(split_gpu_list[rank], args.model_path, args.adapter_path, data_subsets[rank], dataset_name, max_context_length, args.chat_template, return_list, prompts_type))
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
    command = f'python ../eval.py --pred_path={pred_dir}'

    # 执行命令
    exit_code = os.system(command)

    # 检查命令是否成功执行
    if exit_code == 0:
        print("命令执行成功！")
    else:
        print(f"命令执行失败，退出码：{exit_code}")


        # python ../eval.py --pred_path="/data/zecheng/LCM_Stack/evaluation/longbench_v2/scripts/results/Llama3.1-sft-longbench/vanilla"

        # python eval.py --pred_path="/data/zecheng/LCM_Stack/evaluation/longbench_v2/scripts/results/Llama3.1-sft-longbench/vanilla"