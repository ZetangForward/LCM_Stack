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

    all_files = os.listdir(pred_path)
    # output = {
    #     "Easy":[],
    #     "Hard":[],
    #     "Short":[],
    #     "Medium":[],
    #     "Overall":[]
    # }

    output = {}
    # easy, hard, short, medium ,overall= 0, 0, 0, 0, 0
    # easy_acc, hard_acc, short_acc, medium_acc, overall_acc = 0, 0, 0, 0, 0

    for file_name in all_files:
        if ".jsonl" not in file_name:continue
        task = file_name.split('.')[0]

        
        file_path = os.path.join(pred_path, file_name)
        datas = auto_read_data(file_path)
        
        output[file_name] = ([0, 0, 0, 0, 0], 
                             [0, 0, 0, 0, 0])

        for pred in datas:
            
            acc = int(pred['judge'])

            if pred["difficulty"] == "easy":
                # easy += 1
                # easy_acc += acc
                
                output[file_name][0][0] += acc
                output[file_name][1][0] += 1
                
            else:
                # hard += 1
                # hard_acc += acc

                output[file_name][0][1] += acc
                output[file_name][1][1] += 1
            


            if pred['length'] == "short":
                # short += 1
                # short_acc += acc
                
                output[file_name][0][2] += acc
                output[file_name][1][2] += 1


            elif pred['length'] == "medium":
                # medium += 1
                # medium_acc += acc

                output[file_name][0][3] += acc
                output[file_name][1][3] += 1

            # overall += 1
            # overall_acc += acc

            output[file_name][0][4] += acc
            output[file_name][1][4] += 1

            assert pred["length"].lower() != "long"
        


    overall_list = ([0, 0, 0, 0, 0], 
                    [0, 0, 0, 0, 0])

    for k in output.values():
        for i in range(len(k[0])):
            overall_list[0][i] += k[0][i]
            overall_list[1][i] += k[1][i]
            
    output["Overall"] = overall_list

    # print(output)
    result = {k:[] for k in output.keys()}
    for k in output.keys():
        for i in range(5):
            result[k].append(output[k][0][i]/max(1, output[k][1][i]))
            # = [output[k][0][i]/max(1, output[k][1][i]) for i in range(len(output[k][0]))]
            
        # name = '.'.join(file_name.split('.')[:-1])
    pd.DataFrame(result).to_csv(f"{pred_path}/result.csv")

    # python eval.py --pred_path="/data/zecheng/LCM_Stack/evaluation/longbench_v2/results/Llama3.1-sft-longbench/vanilla"

if __name__ == "__main__":
    Fire(main)