import os
import json
import argparse
import numpy as np
from fire import Fire
from modelzipper.tutils import *

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

def main(pred_path: str = None, benchmark_dataset_path: str = None):
    scores = {"index": [],
              "length":[],
              "task":[],
              'pred':[],
              'golden':[],
              "score":[]}
    all_files = os.listdir(pred_path)
    output = ["Model\tOverall\tEasy\tHard\tShort\tMedium\tLong"]
    easy, hard, short, medium, long = 0, 0, 0, 0, 0
    easy_acc, hard_acc, short_acc, medium_acc, long_acc = 0, 0, 0, 0, 0
    all_datas_l = 0
    for file_name in all_files:
        if ".jsonl" not in file_name:continue
        task = file_name.split('.')[0]
        

        file_path = os.path.join(pred_path, file_name)
        # print(file_path)
        datas = auto_read_data(file_path)
        all_datas_l += len(datas)
        for pred in datas:
            
            acc = int(pred['judge'])

            if pred["difficulty"] == "easy":
                easy += 1
                easy_acc += acc
            else:
                hard += 1
                hard_acc += acc

            if pred['length'] == "short":
                short += 1
                short_acc += acc
            elif pred['length'] == "medium":
                medium += 1
                medium_acc += acc
            else:
                long += 1
                long_acc += acc

    easy = max(easy, 1)
    hard = max(hard, 1)
    short = max(short, 1)
    medium = max(medium, 1)
    long = max(long, 1)
            
        # name = '.'.join(file_name.split('.')[:-1])
    output.append("overall"+'\t'+str(round(100*(easy_acc+hard_acc)/all_datas_l, 1))+'\t'+str(round(100*easy_acc/easy, 1))+'\t'+str(round(100*hard_acc/hard, 1))+'\t'+str(round(100*short_acc/short, 1))+'\t'+str(round(100*medium_acc/medium, 1))+'\t'+str(round(100*long_acc/long, 1)))

    
    open(f'{pred_path}/result.txt', 'w', encoding='utf-8').write('\n'.join(output))
    





if __name__ == '__main__':
    Fire(main)
    log_c("evaluation finish")
