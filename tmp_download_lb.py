from datasets import load_dataset

# datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", "trec", \
            # "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
datasets = ["musique"]
for dataset in datasets:
    data = load_dataset('THUDM/LongBench', f"{dataset}", split='test')