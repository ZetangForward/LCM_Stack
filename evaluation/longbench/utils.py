from rouge_score import rouge_scorer
from metrics import (
    qa_f1_score, rouge_score, classification_score, 
    retrieval_score, count_score, code_sim_score,
)


"""
LONG BENCH DATA SETTING
"""
LB_DATA_PROMPT = {
    "narrativeqa_e": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "qasper_e": "You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "multifieldqa_en_e": "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "hotpotqa_e": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "2wikimqa_e": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "musique": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "gov_report_e": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:",
    "qmsum_e": "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:",
    "multi_news_e": "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:",
    "trec_e": "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}",
    "triviaqa_e": "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n{input}",
    "samsum_e": "Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n\n{input}",
    "passage_count_e": "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please carefully read these paragraphs and determine how many unique paragraphs there are after removing duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n{context}\n\nPlease enter the final count of unique paragraphs after removing duplicates. The output format should only contain the number, such as 1, 2, 3, and so on.\n\nThe final answer is: ",
    "passage_retrieval_en_e": "Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc.\n\nThe answer is: ",
    "lcc_e": "Please complete the code given below. \n{context}Next line of code:\n",
    "repobench-p_e": "Please complete the code given below. \n{context}{input}Next line of code:\n"
}

LB_PRED_LEN = {
    "narrativeqa_e": 128,
    "qasper_e": 128,
    "multifieldqa_en_e": 64,
    "hotpotqa_e": 32,
    "2wikimqa_e": 32,
    "musique": 32,
    "qmsum_e": 512,
    "gov_report_e": 512,
    "multi_news_e": 512,
    "trec_e": 64,
    "triviaqa_e": 32,
    "samsum_e": 128,
    "passage_count_e": 32,
    "passage_retrieval_en_e": 32,
    "lcc_e": 64,
    "repobench-p_e": 64
}

DATASET2MAXNEWTOKENS = {
    "narrativeqa_e": 128,
    "narrative_qa_e": 128,
    "qasper_e": 128,
    "multifieldqa_en_e": 64,
    "multifieldqa_zh_e": 64,
    "hotpotqa_e": 32,
    "2wikimqa_e": 32,
    "musique": 32,
    "dureader_e": 128,
    "gov_report_e": 512,
    "qmsum_e": 512,
    "multi_news_e": 512,
    "vcsum_e": 512,
    "trec_e": 64,
    "triviaqa_e": 32,
    "samsum_e": 128,
    "lsht_e": 64,
    "passage_count_e": 32,
    "passage_retrieval_en_e": 32,
    "passage_retrieval_zh_e": 32,
    "lcc_e": 64,
    "repobench-p_e": 64,
    "summ_screen_fd_e":512,
    "squality_e":128,
    "quality_e":128,
    "space_digest_e":128,
    "book_sum_sort_e":128
}

DATASET2CATEGORY = {
    "narrativeqa_e": "EN Single-Doc QA",
    "qasper_e": "EN Single-Doc QA",
    "multifieldqa_en_e": "EN Single-Doc QA",
    "multifieldqa_zh": "CN Single-Doc QA",
    "hotpotqa_e": "EN Multi-Doc QA",
    "2wikimqa_e": "EN Multi-Doc QA",
    "musique": "EN Multi-Doc QA",
    "dureader": "CN Multi-Doc QA",
    "gov_report_e": "EN Summarization",
    "qmsum_e": "EN Summarization",
    "multi_news_e": "EN Summarization",
    "vcsum": "CN Summarization",
    "trec_e": "EN Few-Shot Learning",
    "triviaqa_e": "EN Few-Shot Learning",
    "samsum_e": "EN Few-Shot Learning",
    "lsht": "CN Few-Shot Learning",
    "passage_retrieval_en_e": "EN Synthetic Task",
    "passage_count_e": "EN Synthetic Task",
    "passage_retrieval_zh": "CN Synthetic Task",
    "lcc_e": "Code Completion",
    "repobench-p_e": "Code Completion",
}

DATASET2METRICS = {
    "narrativeqa_e": qa_f1_score,
    "qasper_e": qa_f1_score,
    "multifieldqa_en_e": qa_f1_score,
    "hotpotqa_e": qa_f1_score,
    "2wikimqa_e": qa_f1_score,
    "musique": qa_f1_score,
    "gov_report_e": rouge_score,
    "multi_news_e": rouge_score,
    "trec_e": classification_score,
    "triviaqa_e": qa_f1_score,
    "samsum_e": rouge_score,
    "passage_retrieval_en_e": retrieval_score,
    "passage_count_e": count_score,
    "lcc_e": code_sim_score,
    "repobench-p_e": code_sim_score,
    "qmsum_e": rouge_score,
}


# ALL_LB_TESTING_SETS = [
#     "qasper_e", "multifieldqa_en_e", "hotpotqa_e", "2wikimqa_e", "gov_report_e", 
#     "multi_news_e", "musique_e", "trec_e", "triviaqa_e",  "samsum_e", "passage_count_e", 
#     "passage_retrieval_en_e", "lcc_e", "repobench-p_e", "narrativeqa_e", "qmsum_e",
# ]

ALL_LB_TESTING_SETS = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
ALL_LB_TESTING_SETS = [f"{i}_e" for i in ALL_LB_TESTING_SETS]
ALL_LB_TESTING_SETS.append("musique")

ALL_ZERO_TESTING_SETS = [
    "book_sum_sort", "narrative_qa", "qmsum", "squality", "gov_report", 
    "qasper", "quality", "summ_screen_fd", "musique",  "space_digest",
]


class CustomScorer:
    def __init__(self, real_answer, forbidden_strings):
        self.real_answer = real_answer
        self.forbidden_strings = forbidden_strings
        self.scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
        self.penalty_factor = 100 / (len(self.forbidden_strings) + 1)

    def score(self, response):
        # Calculate initial ROUGE-1 recall score
        initial_scores = self.scorer.score(self.real_answer, response)
        initial_recall_score = initial_scores['rouge1'].recall * 100
        
        # Calculate forbidden strings penalty
        forbidden_penalty_nums = self.count_forbidden_entity_nums(response)
        penalty_value = forbidden_penalty_nums * self.penalty_factor

        # Calculate length penalty
        # length_penalty = self.calculate_length_penalty(response)
        
        # Apply penalties to the ROUGE-1 recall score
        penalized_recall_score = initial_recall_score - penalty_value
        
        return initial_recall_score, penalized_recall_score

    def calculate_length_penalty(self, response):
        # Define a base penalty factor for length
        length_factor = 0.01
        response_length = len(response.split())
        length_penalty = length_factor * response_length
        return min(1, length_penalty)  # Ensure the penalty is between 0 and 1

    def count_forbidden_entity_nums(self, response):
        penalty_count = 0
        
        for forbidden in self.forbidden_strings:
            if forbidden.lower() in response.lower():
                penalty_count += 1
        
        return penalty_count