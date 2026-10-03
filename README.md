# CoV-RankCoT

**Answer-Verification-Driven Retrieval-Augmented Generation for Drilling and Completion Question Answering**

This repository provides the implementation of **CoV-RankCoT**, including multidimensional confidence evaluation, query defect diagnosis, adaptive query rewriting, secondary retrieval, and RankCoT-based knowledge refinement.

## Repository Structure

```text
CoV-RankCoT/
├── Knowledge_Refinement/
│   ├── CoTdata_generation/
│   │   ├── queryCoT_to_answer0.py
│   │   ├── querypassage_to_CoT0.py
│   │   └── template.py
│   │
│   └── answer_generation/
│       ├── evaluate.py
│       ├── queryCoT_to_answer1.py
│       ├── querypassage_to_CoT1.py
│       └── template.py
│
├── merged_rankcot_model/
├── meta-llamaMeta-Llama-3-8B-Instruct/
├── scripts/
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

The module evaluates the initial answer from multiple dimensions, including question consistency, evidence support, and factual and logical consistency.

Low-confidence samples are passed to the query diagnosis and query optimization stages.

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

The corresponding rewriting strategies are:

- semantic disambiguation;
- information completion;
- terminology normalization;
- conservative expansion.

The rewritten query is used for secondary retrieval, while the original query is retained as the user-intent constraint for the subsequent knowledge refinement and answer generation stages.

## RankCoT Knowledge Refinement

The knowledge refinement component follows the RankCoT implementation.

Generate refined CoT knowledge:

```bash
python Knowledge_Refinement/answer_generation/querypassage_to_CoT1.py \
  --model_path merged_rankcot_model \
  --data_path nq_low_confidence_retrieved_GPT.jsonl \
  --output_name nq_low_confidence_retrieved_GPT_COT.jsonl \
  --max_psg_length 1500
```

Generate answers from the refined knowledge:

```bash
python Knowledge_Refinement/answer_generation/queryCoT_to_answer1.py \
  --model_path merged_rankcot_model \
  --data_path nq_low_confidence_retrieved_GPT_COT.jsonl \
  --output_name nq_low_confidence_retrieved_answer_GPT_all.jsonl
```

The paths of the model checkpoint and input/output files can be modified according to the local environment.

## Full Pipeline

Run the main CoV-RankCoT modules sequentially:

```bash
python confidence_evaluation.py
python query_diagnosis.py --mode diagnose
python query_rewrite.py
```

Or use the unified entry:

```bash
python main.py
```

The RankCoT-based knowledge refinement and final answer generation stages can then be executed using the commands provided above.

## Data

The public data used by the official RankCoT implementation can be obtained from the following link:

| Resource | Link |
| --- | --- |
| RankCoT Data | [Google Drive](https://drive.google.com/drive/folders/1QJ63-90RIdjyKwAdCMZKLz5KiFfxEkoq?usp=sharing) |

The downloaded RankCoT data can be organized according to the original RankCoT repository.

## Models and Upstream Repositories

The implementation uses or refers to the following open-source models and repositories:

| Model / Method | Repository |
| --- | --- |
| Llama3-8B-Instruct | [`meta-llama/llama3`](https://github.com/meta-llama/llama3) |
| MiniCPM3-4B | [`OpenBMB/MiniCPM`](https://github.com/OpenBMB/MiniCPM) |
| Qwen2.5-14B-Instruct | [`QwenLM/Qwen2.5`](https://github.com/QwenLM/Qwen2.5) |
| MacBERT-base | [`ymcui/MacBERT`](https://github.com/ymcui/MacBERT) |
| BGE-M3 | [`FlagOpen/FlagEmbedding`](https://github.com/FlagOpen/FlagEmbedding) |
| RankCoT | [`NEUIR/RankCoT`](https://github.com/NEUIR/RankCoT) |

Please follow the licenses and usage requirements of the corresponding upstream repositories.

## Notes

Before running the experiments, please check the following settings:

- model checkpoint paths;
- input and output data paths;
- retrieval corpus configuration;
- MacBERT checkpoint;
- BGE-M3 configuration;
- large language model API configuration;
- dependencies listed in `requirements.txt`.

Model checkpoints and large-scale datasets may need to be downloaded separately according to the corresponding upstream licenses and access requirements.

Do not commit API keys or other private credentials to the repository.

## License

No license has currently been selected for the newly implemented components of this repository.

Third-party models, datasets, and code remain subject to their respective original licenses and usage requirements. Please review and comply with the corresponding upstream licenses before redistribution or public release.
