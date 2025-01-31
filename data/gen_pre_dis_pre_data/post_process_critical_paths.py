import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
print(sys.path)
from vllm import LLM, SamplingParams
from modelzipper.tutils import *
import fire
import datasets
from utils.call_llm_api import call_with_messages, api_client, MODEL_ENDPOINT
import multiprocessing
from datasets import Dataset

class PostProcessor:
    def __init__(self, dataset_dirs, dataset_names, output_path, concate_str=' <evidence> ') -> None:
        self.dataset_dirs = dataset_dirs
        self.dataset_names = dataset_names
        self.output_path = output_path
        self.concate_str = concate_str
        self.init_filter()

    def init_filter(self):
        print(f'Loading datasets from {self.dataset_dirs}')
        self.datasets = {}
        with tqdm(total=len(self.dataset_names) * len(self.dataset_dirs), desc='Loading datasets') as pbar:
            for dataset_name in self.dataset_names:
                self.datasets[dataset_name] = {}
                for dataset_cls, dataset_dir in self.dataset_dirs.items():
                    dataset_path = os.path.join(dataset_dir, dataset_name)
                    if os.path.isdir(dataset_path):
                        self.datasets[dataset_name][dataset_cls] = datasets.load_from_disk(dataset_path)
                    else:
                        self.datasets[dataset_name][dataset_cls] = auto_read_data(dataset_path)
                    pbar.update(1)

    
    def judge_function(self, question, pred_1, pred_2, label):
        system_template = '''below is a question, a golden label and two model prediction results: pred_1 and pred_2, please judge:
        1. Whether the pred_1 is correct with label, return "Correct" or "Incorrect"
        2. Whether the pred_2 is correct with label, return  "Correct" or "Incorrect"
        3. Whether pred_1 is better than pred_2, which is more consistent with the golden label, return "Yes" or "No".
        
        Below is example 1:
        ##Question: Who is Leah?
        ##Label: Leah is Dinah Lenney's mother, whose influence and relationship with Dinah are explored in various essays. Leah's character is depicted as strong, opinionated, and a significant figure in Dinah's life. Dinah reflects on her complex relationship with her mother, touching on themes of approval, identity, and the impact of parental relationships on one's sense of self. Through her reflections on Leah, Dinah navigates the intricacies of mother-daughter relationships, highlighting the ways in which they shape our lives and perceptions.
        ##Pred_1: Leah is the mother of the narrator, whose influence and relationship with Dinah are explored in various essays.
        ##Pred_2: Leah is the mother of the narrator.
        ##Judgement: Correct, Correct, No
        Here is the explanation:
        1. Pred_1 is correct with the label, which is semantic similar.
        2. Pred_2 is correct with the label.
        3. Pred_1 is not better than pred_2, they are the same.
        
        Below is example 2:
        ##Question: Please summarize the biography of Jeff.
        ##Label: Jeff is Elizabeth's brother who unexpectedly shows up at her apartment with her ex-boyfriend, Danny. Jeff's visit is unplanned and brings turmoil into Elizabeth's life as he and Danny need money to flee the country due to Danny's involvement in a crime. Jeff's presence forces Elizabeth to confront her past and the complex dynamics of her family and relationships.
        ##Pred_1: Jeff is Lizzy's older brother. He's always chosen Danny over Lizzy and resents the fact that Lizzy got out of town and left him to take care of their mother. He's stressed and exhausted when he arrives at Lizzy's apartment and needs to sleep. He wants to take Danny to Mexico.
        ##Pred_2: Jeff is Lizzy's older brother, in his 30s. He has a troubled past and has always chosen Danny over Lizzy. He resents Lizzy for leaving him and their mother behind. He is now seeking Lizzy out for money to help Danny, who is a violent and dangerous individual.
        ##Judgement: Incorrect, Incorrect, Yes
        Here is the explanation:
        1. Pred_1 is incorrect with the label.
        2. Pred_2 is incorrect with the label.
        3. Pred_1 is better than pred_2, they are both incorrect, but pred_1 is more consistent with the golden label with more information.
        
        You will be provided with ##Question, ##Label, ##Pred_1, ##Pred_2, please response the ##Judgement directly. 
        Just response the three questions with "Correct", "Incorrect", "Yes" or "No", and concate them with comma. 
        Do not add any other information in the response, including explanations!
        '''

        query = f'{system_template}\n##Question: {question}\n##Label: {label}\n##Pred_1: {pred_1}\n##Pred_2: {pred_2}\n##Judgement: '
        
        judge_res = call_with_messages(api_client, MODEL_ENDPOINT['doubao-pro-128k'], query, max_attempts=5, max_new_tokens=32)
        raw_judge_res_str = judge_res
        if judge_res is None:
            return None, ''
        else:
            judge_res = judge_res.strip().split('\n')[0].strip()
            if 'judgement' in judge_res.lower():
                judge_res = judge_res.split('##Judgement: ')[-1].strip()
            try:
                judge1, judge2, judge3 = [j.strip() for j in judge_res.split(',')]
                if judge1.lower() == 'correct' and judge2.lower() == 'incorrect' and judge3.lower() == 'yes':
                    return [True, True, True], raw_judge_res_str
                elif judge1.lower() == 'correct' and judge2.lower() == 'incorrect' and judge3.lower() == 'no':
                    return [True, True, False], raw_judge_res_str
                elif judge1.lower() == 'incorrect' and judge2.lower() == 'correct' and judge3.lower() == 'yes':
                    return [True, False, True], raw_judge_res_str
                elif judge1.lower() == 'incorrect' and judge2.lower() == 'correct' and judge3.lower() == 'no':
                    return [True, False, False], raw_judge_res_str
                elif judge1.lower() == 'incorrect' and judge2.lower() == 'incorrect' and judge3.lower() == 'yes':
                    return [False, False, True], raw_judge_res_str
                else:
                    return [False, False, False], raw_judge_res_str
            except:  # judge failure
                return None, raw_judge_res_str
        
        return None, raw_judge_res_str
    
    
    def process_chunk(self, data_chunk):
        partial_result = dict(all_ref_text = [], combined_question = [], final_answer = [], label = [], prefix_a = [], siffix_a = [])
        with tqdm(total=len(data_chunk), desc=f'Processing chunk, PID is {os.getpid()}') as pbar:
            for item1, item2, item3 in data_chunk:
                if item1['question'] == item2['question'] and item1['question'] == item3['question']:
                    if item1['question'] == item2['question'] and item1['question'] == item3['question']:
                        judge_score = -1 
                        judge_res, judge_res_str = None, None
                       
                        if judge_res is None:
                            judge_score = -1
                        else:
                            if judge_res[-1]:  
                                if all(judge_res[:2]) or not all(judge_res[:2]):  
                                    judge_score = 1
                                elif judge_res[0] and not judge_res[1]: 
                                    judge_score = 2
                                else:
                                    judge_score = 0
                            else:
                                judge_score = 0
                        partial_result['combined_question'].append(item1['question'])
                        partial_result['label'].append(item1['label'])
                        partial_result['final_answer'].append(item1['predict'])
                        partial_result['prefix_a'].append(item2['predict'])
                        partial_result['siffix_a'].append(item3['predict'])
                        partial_result['all_ref_text'].append(item1['context_lst'])
                pbar.update(1)
        return partial_result
    
    
    def merge_results(self, results):
        final_result = dict(all_ref_text = [], combined_question = [], final_answer = [], label = [], prefix_a = [], siffix_a = [])
        for result in results:
            for key in final_result:
                final_result[key].extend(result[key])
        return final_result
    

    def mp_process_dataset(self, dataset_name, save_path, num_process=32):
        all_samples = self.datasets[dataset_name]
        pred_w_full_paths = all_samples['pred_w_full_paths']
        pred_w_half_paths = all_samples['pred_w_half_paths']
        pred_wo_critical_paths = all_samples['pred_wo_critical_paths']
        
        # Prepare data chunks for multiprocessing
        data_chunks = list(zip(pred_w_full_paths, pred_w_half_paths, pred_wo_critical_paths))
        chunk_size = len(data_chunks) // num_process
        data_chunks = [data_chunks[i:i + chunk_size] for i in range(0, len(data_chunks), chunk_size)]
        print("begin to feed results to the API ...")
        # Create a pool of processes
        with multiprocessing.Pool(processes=num_process) as pool:
            # Use imap instead of map to allow for progress tracking
            results = list(tqdm(
                pool.imap(self.process_chunk, data_chunks),
                total=len(data_chunks),
                desc=f"Processing {dataset_name}"
            ))

        # Merge results
        merged_data = self.merge_results(results)
        # Save the merged data
        Dataset.from_dict(merged_data).save_to_disk(save_path, num_proc=32)

        
    def sp_process_dataset(self, dataset_name, save_path):
        '''single process dataset'''
        all_samples = self.datasets[dataset_name]
        merged_data = dict(
            all_ref_text = [],
            combined_question = [],
            final_answer = [],
            label = [],
            prefix_a = [],
            siffix_a = [], 
            judge_scores = [],
            judger_preds = []
        )
        pred_w_full_paths = all_samples['pred_w_full_paths']
        pred_w_half_paths = all_samples['pred_w_half_paths']
        pred_wo_critical_paths = all_samples['pred_wo_critical_paths']
        with tqdm(total=len(pred_w_full_paths), desc=f'Processing {dataset_name}, which has {len(merged_data["judge_scores"])} items') as pbar:
            for item1, item2, item3 in zip(pred_w_full_paths, pred_w_half_paths, pred_wo_critical_paths):
                if item1['question'] == item2['question'] and item1['question'] == item3['question']:
                    judge_score = -1  
                    judge_res = None
                    if judge_res is None:
                        judge_score = -1
                    else:
                        if judge_res[-1]:  
                            if all(judge_res[:2]) or not all(judge_res[:2]): 
                                judge_score = 1
                            elif judge_res[0] and not judge_res[1]: 
                                judge_score = 2
                            else:
                                judge_score = 0
                        else:
                            judge_score = 0
                    merged_data['combined_question'].append(item1['question'])
                    merged_data['label'].append(item1['label'])
                    merged_data['final_answer'].append(item1['predict'])
                    merged_data['prefix_a'].append(item2['predict'])
                    merged_data['siffix_a'].append(item3['predict'])
                    merged_data['all_ref_text'].append(item1['context_lst'])
                pbar.update(1)
        print(f'judge dataset: {dataset_name} finish... has length', len(merged_data))
        Dataset.from_dict(merged_data).save_to_disk(save_path)


    def process_all_datasets(self):
        for dataset_name in self.dataset_names:
            save_path = os.path.join(self.output_path, dataset_name)
            self.mp_process_dataset(dataset_name, save_path, num_process=48)
        print('All datasets have been processed!')


if __name__ == '__main__':
    data_dir = '/data/'
    save_dir = '/data/pre-process'
    classification_dir = {
        'pred_w_full_paths': os.path.join(data_dir, 'pred_w_full_paths'),
        'pred_w_half_paths': os.path.join(data_dir, 'pred_w_half_paths'),
        'pred_wo_critical_paths': os.path.join(data_dir, 'pred_wo_critical_paths')
    }
    
    critical_data_names = ['gpt-bio_book', 'gpt-multi_detail_paper_short', 'longalpaca-train', 'gpt-multi_detail_book', 'gpt-one_detail_paper', 'gpt-multi_detail_paper_long']
    
    processer = PostProcessor(
        classification_dir, 
        critical_data_names, 
        output_path=save_dir,
    )
    
    print('Begin to process all datasets ...')

    processer.process_all_datasets()
    
    print(f'All datasets have been processed! The results are saved in {save_dir}!')