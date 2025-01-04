import os
import json
import argparse
import numpy as np
from fire import Fire
from modelzipper.tutils import *
from loguru import logger
from utils import DATASET2METRICS, ALL_LB_TESTING_SETS, DATASET2MAXNEWTOKENS
from datasets import load_dataset

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    return parser.parse_args(args)

import os
import json
import argparse
import numpy as np


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    return parser.parse_args(args)

def scorer_e(dataset, predictions, answers, lengths, all_classes):
    scores = {"0-4k": [], "4-8k": [], "8k+": []}
    for (prediction, ground_truths, length) in zip(predictions, answers, lengths):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, DATASET2METRICS[dataset](prediction, ground_truth, all_classes=all_classes))
        if length < 4000:
            scores["0-4k"].append(score)
        elif length < 8000:
            scores["4-8k"].append(score)
        else:
            scores["8k+"].append(score)
    for key in scores.keys():
        scores[key] = round(100 * np.mean(scores[key]), 2)
    return scores

def scorer(dataset, predictions, answers, all_classes):
    total_score = 0.
    for (prediction, ground_truths) in zip(predictions, answers):
        score = 0.
        if '[/INST]' in prediction:
            prediction = prediction.replace('[/INST]', '')
        if dataset in ["trec", "triviaqa_e", "samsum_e", "lsht", 'narrativeqa', '']:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, DATASET2METRICS[dataset](prediction, ground_truth, all_classes=all_classes))
        total_score += score
    return round(100 * total_score / len(predictions), 2)


def main(pred_path: str = None, benchmark_dataset_path: str = None):
    scores = dict()
    all_files = os.listdir(pred_path)

    ## extract all golden data classes
    data_classes = {}
    if benchmark_dataset_path is None:
        for task_name in ALL_LB_TESTING_SETS:
            content = load_dataset('THUDM/LongBench', task_name, split='test')
            data_classes[task_name] = content[0]["all_classes"]
    else:
        all_benchmark_data_files = os.listdir(benchmark_dataset_path)
        for f in all_benchmark_data_files:
            task_name = f.split('.')[0]
            content = auto_read_data(os.path.join(benchmark_dataset_path, f))
            data_classes[task_name] = content[0]["all_classes"]
    
    ## read all predicted datasets
    for filename in tqdm(all_files):
        if not filename.endswith("jsonl"):
            continue
        dataset_name = filename.split('.')[0]
        pred_dataset_length, all_classes = DATASET2MAXNEWTOKENS[dataset_name], data_classes[dataset_name]
        predictions, answers, lengths = [], [], []
       
        with open(os.path.join(pred_path, filename), "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                predictions.append(data["pred_str"])
                answers.append(data["answers"])
                lengths.append(data["length"])

        if len(predictions) == 0: continue
        score = scorer_e(dataset_name, predictions, answers, lengths, all_classes)
        scores[dataset_name] = score

    # 定义列顺序
    columns = [
        ['Single-Document QA', 'Single-Document QA', 'Single-Document QA',
        'Multi-Document QA', 'Multi-Document QA', 'Multi-Document QA', 'Multi-Document QA',
        'Summarization', 'Summarization', 'Summarization', 'Summarization',
        'Few-shot Learning', 'Few-shot Learning', 'Few-shot Learning', 'Few-shot Learning',
        'Synthetic Tasks', 'Synthetic Tasks', 'Synthetic Tasks',
        'Code Completion', 'Code Completion', 'Code Completion', 'ALL Avg.'
        ],
        ['qasper_e', 'multifieldqa_en_e', 'Avg.',
        'hotpotqa_e', '2wikimqa_e', 'musique', 'Avg.',
        'gov_report_e', 'qmsum_e', 'multi_news_e', 'Avg.',
        'trec_e', 'triviaqa_e', 'samsum_e', 'Avg.',
        'passage_count_e', 'passage_retrieval_en_e', 'Avg.',
        'lcc_e', 'repobench-p_e', 'Avg.', ''
        ]
    ]

    
    df = pd.DataFrame(scores) # 将 scores 转换为 DataFrame
    average_row = df.mean(axis=0).rename("Avg.")
    
    df = pd.concat([df, average_row.to_frame().T]) # 将平均值作为新行添加到 DataFrame
    df = df.round(2) # 保留两位小数 

    # 创建多级列索引
    multi_index = pd.MultiIndex.from_tuples(zip(*columns))

    # 创建新的 DataFrame 按照指定列顺序
    sorted_df = pd.DataFrame(columns=multi_index)

    # 填充数据
    for col_0, col_1 in zip(columns[0], columns[1]):
        if col_1 in df.columns:
            sorted_df[(col_0, col_1)] = df[col_1]
        else:
            sorted_df[(col_0, col_1)] = 0  # 填充缺失列为 NaN
    
    out_path = os.path.join(pred_path, "result.csv")
    # 保存为 CSV 文件
    sorted_df.to_csv(out_path, index=False)


if __name__ == '__main__':
    Fire(main)
    log_c("evaluation finish")
