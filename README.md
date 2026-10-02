# CoV-RankCoT

**Answer-Verification-Driven Retrieval-Augmented Generation for Drilling and Completion Question Answering**

This repository provides the main implementation of **CoV-RankCoT**, an answer-verification-driven retrieval-augmented generation framework for drilling and completion question answering.

CoV-RankCoT extends the RankCoT knowledge refinement framework with three main components:

- domain-rule-constrained multidimensional confidence evaluation;
- MacBERT-based query defect diagnosis;
- category-guided adaptive query optimization.

The framework first refines retrieved evidence and generates an initial answer. The initial answer is then evaluated through a multidimensional confidence mechanism. If the confidence score satisfies the predefined threshold, the answer is directly returned. Otherwise, the original query is diagnosed by a fine-tuned MacBERT model and rewritten according to the predicted defect category, followed by secondary retrieval and knowledge refinement.



## Repository Structure

```text
CoV-RankCoT/
├── Knowledge_Refinement/
│   └── RankCoT-related knowledge refinement components
├── confidence_evaluation.py
├── query_diagnosis.py
├── query_rewrite.py
├── main.py
├── requirements.txt
└── README.md
```

The main files are:

- `confidence_evaluation.py`  
  Implements the domain-rule-constrained multidimensional confidence evaluation mechanism.

- `query_diagnosis.py`  
  Implements MacBERT-based query defect diagnosis.

- `query_rewrite.py`  
  Implements category-guided query rewriting and secondary retrieval.

- `main.py`  
  Provides the main execution entry for the proposed framework.

- `Knowledge_Refinement/`  
  Contains the RankCoT-related knowledge refinement implementation.

- `requirements.txt`  
  Lists the main Python dependencies used in the experiments.

---

## Environment

The main experimental environment is:

```text
Python 3.12
PyTorch 2.8
CUDA 12.8
Transformers
FlagEmbedding
OpenAI
scikit-learn
```

Install the required dependencies:

```bash
python -m pip install -r requirements.txt
```

If an external large language model API is used, configure the API key through an environment variable:

```bash
export OPENAI_API_KEY="YOUR_API_KEY"
```

Do not commit private API keys or credentials to the public repository.

---

## Quick Start

The main modules can be executed sequentially.

### 1. Multidimensional Confidence Evaluation

Run:

```bash
python confidence_evaluation.py
```

The confidence evaluation module calculates three scores:

- question consistency;
- evidence support;
- factual and logical consistency.

The framework further estimates query-type probabilities and dynamically adjusts the weights of the three evaluation dimensions.

The final confidence score is used to determine whether the initial answer should be directly returned or whether query optimization should be triggered.

For a confidence threshold `tau = 8.0`:

```text
epsilon >= 8.0
    -> return the initial answer

epsilon < 8.0
    -> activate query defect diagnosis and adaptive query optimization
```

---

## Multidimensional Confidence Evaluation

The multidimensional confidence score contains three components.

### Question Consistency

This dimension evaluates whether the generated answer is consistent with the semantic intent of the original query.

```text
S1 = Question Consistency Score
```

### Evidence Support

This dimension evaluates whether the retrieved evidence sufficiently supports the major factual claims in the generated answer.

```text
S2 = Evidence Support Score
```

### Factual and Logical Consistency

This dimension evaluates factual correctness and logical consistency by jointly considering retrieved evidence and relevant drilling and completion domain rules.

```text
S3 = Factual and Logical Consistency Score
```

The final confidence score is calculated using query-type-aware dynamic weights:

```text
epsilon = w1 * S1 + w2 * S2 + w3 * S3
```

The weight adaptation coefficient is set to:

```text
eta = 0.10
```

---

## Query Defect Diagnosis

Low-confidence samples are passed to the MacBERT-based query defect diagnosis module.

Train the query defect classifier:

```bash
python query_diagnosis.py --mode train
```

Diagnose low-confidence queries:

```bash
python query_diagnosis.py --mode diagnose
```

The query defect classifier contains four categories:

1. semantic ambiguity;
2. semantic information missing;
3. non-standard terminology;
4. no obvious expression defect.

The classifier is initialized from `MacBERT-base` and fine-tuned using full-parameter supervised learning.

The model output contains the predicted category and the probability distribution over the four classes.

Example:

```json
{
  "predicted_label": "term_nonstandard",
  "predicted_label_cn": "术语表达不规范",
  "probabilities": {
    "semantic_ambiguity": 0.0312,
    "semantic_missing": 0.0518,
    "term_nonstandard": 0.8936,
    "no_obvious_defect": 0.0234
  }
}
```

---

## Category-Guided Query Optimization

Run:

```bash
python query_rewrite.py
```

The four query defect categories are mapped to four rewriting strategies:

| Query Defect | Rewriting Strategy |
| --- | --- |
| Semantic ambiguity | Semantic disambiguation |
| Semantic information missing | Information completion |
| Non-standard terminology | Terminology normalization |
| No obvious expression defect | Conservative expansion |

The rewritten query is used for secondary retrieval.

The original query is retained as an intent constraint in the subsequent knowledge refinement and answer generation stages to reduce semantic drift.

The query optimization module therefore follows:



## RankCoT-Based Knowledge Refinement

The knowledge refinement component is based on the official RankCoT implementation.

Official RankCoT repository:

https://github.com/NEUIR/RankCoT

The RankCoT-based inference process contains two major stages.

### Stage 1: Query-Passage to Refined CoT

Use the trained RankCoT model to refine the retrieved evidence:

```bash
python src/answer_generation/querypassage_to_CoT.py \
  --model_path /root/autodl-tmp/merged_rankcot_model \
  --data_path nq_low_confidence_retrieved_GPT.jsonl \
  --output_name nq_low_confidence_retrieved_GPT_COT.jsonl \
  --max_psg_length 1500
```

This stage generates refined reasoning knowledge from the query and retrieved passages.

The output file is:

```text
nq_low_confidence_retrieved_GPT_COT.jsonl
```

---

### Stage 2: Refined CoT to Answer

Generate answers using the refined knowledge:

```bash
python src/answer_generation/queryCoT_to_answer.py \
  --model_path /root/autodl-tmp/merged_rankcot_model \
  --data_path /root/autodl-tmp/nq_low_confidence_retrieved_GPT_COT.jsonl \
  --output_name /root/autodl-tmp/nq_low_confidence_retrieved_answer_GPT_all.jsonl
```

The final answer file is:

```text
/root/autodl-tmp/nq_low_confidence_retrieved_answer_GPT_all.jsonl
```

For other datasets, modify `--data_path` and `--output_name` accordingly.

---

## Full Pipeline

The proposed components can be executed sequentially:

```bash
python confidence_evaluation.py

python query_diagnosis.py --mode diagnose

python query_rewrite.py
```

Alternatively, use the unified entry point:

```bash
python main.py
```

The complete workflow is:

```text
Initial Retrieval
    |
    v
RankCoT Knowledge Refinement
    |
    v
Initial Answer
    |
    v
confidence_evaluation.py
    |
    +-----------------------------+
    |                             |
High Confidence             Low Confidence
    |                             |
    v                             v
Final Output              query_diagnosis.py
                                  |
                                  v
                           query_rewrite.py
                                  |
                                  v
                         Secondary Retrieval
                                  |
                                  v
                   RankCoT Knowledge Refinement
                                  |
                                  v
                              Final Answer
```

---

## Intermediate Files

The major intermediate files can be organized as follows:

```text
Initial answers
    |
    v
low_confidence_multidim.jsonl
    |
    v
low_confidence_with_diagnosis.jsonl
    |
    v
low_confidence_rewrite_retrieved.jsonl
    |
    v
nq_low_confidence_retrieved_GPT_COT.jsonl
    |
    v
nq_low_confidence_retrieved_answer_GPT_all.jsonl
```

These files correspond to:

1. low-confidence sample selection;
2. query defect diagnosis;
3. adaptive query rewriting and secondary retrieval;
4. RankCoT knowledge refinement;
5. final answer generation.

---

## Data

The experiments use the following datasets:

- Natural Questions (NQ);
- HotpotQA;
- TriviaQA;
- Hq-Oil.

`Hq-Oil` is a drilling and completion domain question-answering dataset constructed for this study.

Public benchmark datasets should be obtained according to their original distribution licenses.

The drilling and completion domain data contain desensitized engineering materials and related domain resources. Therefore, the current repository mainly releases the implementation and experimental workflow, while the original domain data are not redistributed.

Users may replace the domain corpus and question-answering data with their own datasets following the expected input format.

---

## Input Data Format

A typical retrieval input sample can be represented as:

```json
{
  "id": "sample_001",
  "query": "What is the current bottom-hole flowing pressure of the well?",
  "passages": [
    {
      "segment": "Retrieved passage 1"
    },
    {
      "segment": "Retrieved passage 2"
    }
  ]
}
```

A low-confidence sample after confidence evaluation may contain:

```json
{
  "query": "Example query",
  "confidence_evaluation": {
    "epsilon": 7.42,
    "threshold": 8.0,
    "decision": "low"
  }
}
```

After query diagnosis:

```json
{
  "query_diagnosis": {
    "predicted_label": "term_nonstandard",
    "predicted_label_cn": "术语表达不规范"
  }
}
```

After adaptive query optimization:

```json
{
  "original_query": "Original query",
  "rewritten_query": "Optimized retrieval query",
  "query_optimization": {
    "rewrite_strategy": "Terminology normalization"
  },
  "second_retrieval": {
    "top_k": 5,
    "documents": []
  }
}
```

---

## Main Experimental Settings

The main experimental settings used in the implementation include:

```text
Top-K retrieval passages: 5
Confidence threshold tau: 8.0
Dynamic-weight coefficient eta: 0.10
MacBERT maximum sequence length: 128
MacBERT batch size: 16
MacBERT maximum training epochs: 5
GPT temperature: 0
```

Please modify model paths and dataset paths according to your local environment.

---

## Reproducibility Notes

Before running the experiments, verify the following configurations:

1. RankCoT model checkpoint;
2. dataset path;
3. retrieval corpus path;
4. BGE-M3 configuration;
5. MacBERT checkpoint;
6. large language model API configuration;
7. output directories;
8. Python dependencies in `requirements.txt`.

Example model path:

```text
/root/autodl-tmp/merged_rankcot_model
```

Example data path:

```text
/root/autodl-tmp/nq_low_confidence_retrieved_GPT.jsonl
```

Runtime and memory consumption may vary depending on the hardware environment and external model API latency.

---

## Upstream RankCoT

The knowledge refinement component follows the RankCoT framework:

**RankCoT: Refining Knowledge for Retrieval-Augmented Generation through Ranking Chain-of-Thoughts**

Official repository:

https://github.com/NEUIR/RankCoT

This repository extends the RankCoT-based pipeline with:

- multidimensional answer confidence evaluation;
- drilling and completion domain-rule constraints;
- MacBERT-based query defect diagnosis;
- category-guided query rewriting;
- confidence-triggered secondary retrieval.

Users of RankCoT-related code should follow the license and citation requirements of the original RankCoT repository.

---

## Citation

If this repository is useful for your research, please cite the corresponding paper.

```bibtex
@article{cov_rankcot,
  title   = {Answer-Verification-Driven Retrieval-Augmented Generation for Drilling and Completion Question Answering},
  author  = {Haochang Wang and Zhongyang Li},
  journal = {Computer Applications and Software Research},
  year    = {2026}
}
```

The final bibliographic information should be updated after formal publication.

---

## Acknowledgements

This work builds upon the RankCoT framework.

We thank the authors of RankCoT and the developers of the open-source models and libraries used in this project.

---

## License

Third-party components remain subject to their original licenses.

RankCoT-related code and derived components should follow the license terms of the official RankCoT repository.

The license for the newly implemented components in this repository should be determined by the authors before public release.
