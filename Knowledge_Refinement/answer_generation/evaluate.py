
import pandas as pd
import json
import os
# if os.path.exists("tqa_queryCoT_to_answer.jsonl"):
#     jsonl_file_path = "tqa_queryCoT_to_answer.jsonl"
# if os.path.exists("direct_model_with_answer.jsonl"):
#     jsonl_file_path = "direct_model_with_answer.jsonl"


# if os.path.exists("/root/autodl-tmp/nq_high_confidence_clean_GPT.jsonl"):
#     jsonl_file_path = "/root/autodl-tmp/nq_high_confidence_clean_GPT.jsonl"

# if os.path.exists("nq_low_confidence_retrieved_answer_GPT_all.jsonl"):
#     jsonl_file_path = "nq_low_confidence_retrieved_answer_GPT_all.jsonl"

#作者实验数据集
if os.path.exists("/root/autodl-tmp/nq_queryCoT_to_answer.jsonl"):
    jsonl_file_path = "/root/autodl-tmp/nq_queryCoT_to_answer.jsonl"
#我的实验数据集
if os.path.exists("nq_querypassage_to_answer2.jsonl"):
    jsonl_file_path = "nq_querypassage_to_answer2.jsonl"
#if os.path.exists("Hq-Oil_to_answer.jsonl"):
   # jsonl_file_path = "Hq-Oil_to_answer.jsonl"
##if os.path.exists("hpqa_to_answer.jsonl"):
   # jsonl_file_path = "hpqa_to_answer.jsonl"
# elif os.path.exists("../../tqa_queryCoT_to_answer.jsonl"):
#
#     jsonl_file_path = "../../tqa_queryCoT_to_answer.jsonl"
else:
    raise FileNotFoundError("找不到 nq_queryCoT_to_answer.jsonl 文件")

n = 0
x = 0
# Read JSON Lines file and create a DataFrame
json_lines = []
with open(jsonl_file_path, 'r', encoding='utf-8') as jsonl_file:
    for line in jsonl_file:
        n+=1
        data = json.loads(line)
        ground_truth = data['ground_truth']
        answers = str(data['model_answer'])
        if any(answer.lower() in answers.lower() for answer in ground_truth):
            x+=1

print(x/n)
