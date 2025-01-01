from modelzipper.tutils import *
from utils import ALL_LB_TESTING_SETS, LB_DATA_PROMPT, LB_PRED_LEN, DATASET2CATEGORY
from datasets import load_dataset
import argparse
import torch
import torch.multiprocessing as mp
import numpy as np
from loguru import logger
from peft import PeftModelForCausalLM

context_max_length = {"8k_setting": 7200, "tiny_setting": 15500, "normal_setting": 32000, "long_setting": 63500, "ultra_long_setting": 127500}

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_pred(rank=None, model_path=None, adapter_path=None, datasets=None, dataset_name=None, max_context_length=None, return_list=None):
    
    logger.info(f"rank {rank} is processing {dataset_name} length {len(datasets)} ...")
    # load models
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    test_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, use_flash_attention_2="flash_attention_2", trust_remote_code=True).to(torch.device(f'cuda:{rank}'))
    if adapter_path:
        test_model = PeftModelForCausalLM.from_pretrained(test_model, adapter_path).eval()

    if hasattr(test_model, "generation_config"):
        eos_token_id = test_model.generation_config.eos_token_id
    else:
        eos_token_id = tokenizer.eos_token_id
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
        
    eos_token_id.append(tokenizer.encode("\n", add_special_tokens=False)[-1])
    PROMPT_TEMPLATE, PRED_LENGTH = LB_DATA_PROMPT[dataset_name], LB_PRED_LEN[dataset_name]
    
    pred_res = []
    with torch.no_grad(), tqdm(total=len(datasets)) as pbar:
        for sample in datasets:
            if hasattr(test_model, "memory"):
                test_model.memory.reset()

            context, input_, answers = sample['context'], sample['input'], sample['answers']
            prompt = PROMPT_TEMPLATE.format(input=input_, context=context)

            if not DATASET2CATEGORY[dataset_name] in ["EN Few-Shot Learning", "Code Completion"]:
                prompt = tokenizer.apply_chat_template(
                    [{'role': 'user', 'content': prompt}],
                    add_generation_prompt=True, tokenize=False
                )

            textual_input = tokenizer(prompt, return_tensors="pt").input_ids[0].to(test_model.device)

            max_context_length = max_context_length - PRED_LENGTH - 100 # for chat template
            if len(textual_input) > max_context_length:
                half = int(max_context_length/2)
                prompt = tokenizer.decode(textual_input[:half], skip_special_tokens=True) + tokenizer.decode(textual_input[-half:], skip_special_tokens=True)

            input_ids = tokenizer(prompt, return_tensors="pt").to(test_model.device).input_ids

            if dataset_name in ["2wikimqa_e", "hotpotqa_e", "musique_e", "multifieldqa_en_e", "qasper_e", "narrativeqa_e", "samsum_e"]:
                outputs = test_model.generate(
                    input_ids, 
                    max_new_tokens=PRED_LENGTH, 
                    do_sample=None,
                    begin_suppress_tokens=eos_token_id,
                    eos_token_id=eos_token_id, temperature=None,
                    top_p=None,
                )[0]
            elif dataset_name in ['gov_report_e', 'qmsum_e', 'multi_news_e']:
                outputs = test_model.generate(
                    input_ids,
                    max_new_tokens=PRED_LENGTH,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                )[0]
            else:
                outputs = test_model.generate(
                    input_ids,
                    max_new_tokens=PRED_LENGTH,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                )[0]

            pred_str = tokenizer.decode(outputs[input_ids.shape[-1]:], skip_special_tokens=True)
            pred_res.append({"dataset_name": dataset_name, "pred_str": pred_str, "answers": answers}) 
            pbar.update(1)
            
    return_list.extend(pred_res)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="lb testing")
    parser.add_argument('--model_path', type=str, default=None, help='Path to the model')
    parser.add_argument('--adapter_path', type=str, default=None, help='Path to the PEFT model')
    parser.add_argument('--save_path', type=str, default=None, help='Path to save the output')
    parser.add_argument('--model_max_length_setting', type=str, default="normal_setting", help='Model max length setting')
    parser.add_argument('--seed', type=int, default=27, help='default seed')

    args = parser.parse_args()
    
    world_size = torch.cuda.device_count()
    mp.set_start_method('spawn', force=True)
    
    log_c(f'begin to eval on {world_size} gpus ...')
    
    if args.adapter_path:
        suffix_tag = f"{args.adapter_path.split('/')[-2]}-{args.adapter_path.split('/')[-2]}"
        pred_dir = os.path.join(args.save_path, suffix_tag)
    else:
        pred_dir = os.path.join(args.save_path, "vanilla")
    
    if os.path.exists(pred_dir):
        already_finish_files = auto_read_dir(pred_dir, file_suffix=".jsonl")
        already_finish_files = [os.path.basename(f).split('.')[0] for f in already_finish_files]
        
        # check generated cases
        for f in already_finish_files:
            num_test_cases = len(load_dataset('THUDM/LongBench', f, split='test'))
            num_pred_cases = len(auto_read_data(os.path.join(pred_dir, f + ".jsonl")))
            if num_test_cases != num_pred_cases: 
                print(f"{f} has not been processed, removing it from finished files ...")
                already_finish_files.remove(f)
    else:
        auto_mkdir(pred_dir)
        already_finish_files = []
    
    test_datasets = list(set(ALL_LB_TESTING_SETS) - set(already_finish_files))
    logger.info(f"evaluating on datasets: {test_datasets}")
    max_context_length = context_max_length[args.model_max_length_setting]
    logger.info(f"max_context_length are set as {max_context_length}")
    
    seed_everything(args.seed)

    for dataset_name in test_datasets:
        test_data = load_dataset('THUDM/LongBench', dataset_name, split='test')
        save_res_path = os.path.join(pred_dir, dataset_name + ".jsonl")
        data_subsets = []
        
        for i in range(world_size):
            subset = test_data.shard(num_shards=world_size, index=i)
            data_subsets.append(subset)

        with tqdm(total=world_size) as pbar:
            processes = []
            manager = mp.Manager()
            return_list = manager.list()
             
            for rank in range(world_size):
                p = mp.Process(target=get_pred, args=(rank, args.model_path, args.adapter_path, data_subsets[rank], dataset_name, max_context_length, return_list))
                p.start()
                processes.append(p)
            
            for p in processes:
                p.join()
                pbar.update()
        
        results = list(return_list)

        auto_save_data(results, save_res_path)