# CoV-RankCoT

**Answer-Verification-Driven Retrieval-Augmented Generation for Drilling and Completion Question Answering**

This repository provides the implementation of **CoV-RankCoT**, including multidimensional confidence evaluation, query defect diagnosis, adaptive query rewriting, secondary retrieval, and RankCoT-based knowledge refinement.

## Repository Structure

```text
CoV-RankCoT/
├── Knowledge_Refinement/
├── confidence_evaluation.py
├── query_diagnosis.py
├── query_rewrite.py
├── main.py
├── requirements.txt
└── README.md
```

## Installation

Install the required dependencies:

```bash
python -m pip install -r requirements.txt
```

Main experimental environment:

```text
Python 3.12
PyTorch 2.8
CUDA 12.8
```

## Confidence Evaluation

Run the multidimensional confidence evaluation module:

```bash
python confidence_evaluation.py
```

Low-confidence samples are passed to the query diagnosis and optimization stages.

## Query Defect Diagnosis

Train the MacBERT-based query defect classifier:

```bash
python query_diagnosis.py --mode train
```

Diagnose low-confidence queries:

```bash
python query_diagnosis.py --mode diagnose
```

The classifier contains four query categories:

1. semantic ambiguity;
2. semantic information missing;
3. non-standard terminology;
4. no obvious expression defect.

## Adaptive Query Rewriting

Run category-guided query rewriting and secondary retrieval:

```bash
python query_rewrite.py
```

The four rewriting strategies are:

- semantic disambiguation;
- information completion;
- terminology normalization;
- conservative expansion.

## RankCoT Knowledge Refinement

The knowledge refinement component follows the RankCoT implementation.

Generate refined CoT knowledge:

```bash
python src/answer_generation/querypassage_to_CoT.py \
  --model_path /root/autodl-tmp/merged_rankcot_model \
  --data_path nq_low_confidence_retrieved_GPT.jsonl \
  --output_name nq_low_confidence_retrieved_GPT_COT.jsonl \
  --max_psg_length 1500
```

Generate answers from the refined knowledge:

```bash
python src/answer_generation/queryCoT_to_answer.py \
  --model_path /root/autodl-tmp/merged_rankcot_model \
  --data_path /root/autodl-tmp/nq_low_confidence_retrieved_GPT_COT.jsonl \
  --output_name /root/autodl-tmp/nq_low_confidence_retrieved_answer_GPT_all.jsonl
```

## Full Pipeline

Run the main modules sequentially:

```bash
python confidence_evaluation.py
python query_diagnosis.py --mode diagnose
python query_rewrite.py
```

Or use the unified entry:

```bash
python main.py
```

## Models and Upstream Repositories

The implementation uses or refers to the following open-source models and repositories:

| Model / Method | GitHub Repository |
| --- | --- |
| Llama3-8B-Instruct | `meta-llama/llama3` |
| MiniCPM3-4B | `OpenBMB/MiniCPM` |
| Qwen2.5-14B-Instruct | `QwenLM/Qwen2.5` |
| MacBERT-base | `ymcui/MacBERT` |
| BGE-M3 | `FlagOpen/FlagEmbedding` |
| RankCoT | `NEUIR/RankCoT` |

Please follow the licenses and usage requirements of the corresponding upstream repositories.
