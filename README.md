# CoV-RankCoT
nvironment
Recommended environment:
Python 3.12
PyTorch 2.8
CUDA 12.8
Transformers
FlagEmbedding
OpenAI
scikit-learn

Install the dependencies with:
python -m pip install -r requirements.txt

If an external large language model API is used, configure the API key through an environment variable:
export OPENAI_API_KEY="YOUR_API_KEY"

Do not upload private API keys to a public repository.
Multidimensional Confidence Evaluation
Run:
python confidence_evaluation.py

The confidence evaluation module calculates three scores:
- question consistency;
- evidence support;
- factual and logical consistency.
Query-type probabilities are further used to dynamically determine the weights of the three evaluation dimensions.
Samples whose confidence scores are below the predefined threshold are passed to the query defect diagnosis module.
Query Defect Diagnosis
Train the MacBERT query defect classifier:
python query_diagnosis.py --mode train

Diagnose low-confidence queries:
python query_diagnosis.py --mode diagnose

The query defect classifier contains four categories:
1. semantic ambiguity;
2. semantic information missing;
3. non-standard terminology;
4. no obvious expression defect.
Category-Guided Query Optimization
Run:
python query_rewrite.py

The four query defect categories are mapped to four rewriting strategies:
- semantic disambiguation;
- information completion;
- terminology normalization;
- conservative expansion.
The rewritten query is used for secondary retrieval, while the original query is retained as an intent constraint for the subsequent knowledge refinement and answer generation stages.
RankCoT-Based Knowledge Refinement
The knowledge refinement component is based on the RankCoT implementation.
First, use the trained RankCoT model to refine the retrieved evidence:
python src/answer_generation/querypassage_to_CoT.py \
    --model_path /root/autodl-tmp/merged_rankcot_model \
    --data_path nq_low_confidence_retrieved_GPT.jsonl \
    --output_name nq_low_confidence_retrieved_GPT_COT.jsonl \
    --max_psg_length 1500

Then generate the answer based on the refined knowledge:
python src/answer_generation/queryCoT_to_answer.py \
    --model_path /root/autodl-tmp/merged_rankcot_model \
    --data_path /root/autodl-tmp/nq_low_confidence_retrieved_GPT_COT.jsonl \
    --output_name /root/autodl-tmp/nq_low_confidence_retrieved_answer_GPT_all.jsonl

The first command produces refined reasoning knowledge from the query and retrieved passages. The second command generates the final answer using the refined knowledge.
For other datasets, modify --data_path and --output_name accordingly.
Full Pipeline
The main components can be executed sequentially:
python confidence_evaluation.py
python query_diagnosis.py --mode diagnose
python query_rewrite.py

Alternatively, use the unified entry point:
python main.py

The RankCoT-based knowledge refinement and answer generation stages can then be executed using the commands provided above.
Data
The experiments include NQ, HotpotQA, TriviaQA, and the self-constructed drilling and completion question-answering dataset Hq-Oil.
Public datasets should be obtained according to their original distribution licenses.
Hq-Oil contains desensitized drilling and completion domain resources and related domain materials. Therefore, this repository currently focuses on releasing the implementation and experimental pipeline, while the original domain data are not redistributed.
Users may replace the domain corpus and question-answering data with their own datasets following the expected input formats.
RankCoT
The knowledge refinement module is based on the official RankCoT implementation:
https://github.com/NEUIR/RankCoT
RankCoT:
RankCoT: Refining Knowledge for Retrieval-Augmented Generation through Ranking Chain-of-Thoughts

This repository extends the RankCoT-based pipeline with the components proposed in our work, including multidimensional confidence evaluation, query defect diagnosis, and category-guided adaptive query optimization.
If you use the RankCoT-related components, please follow the license requirements of the original repository and cite the corresponding RankCoT paper.
Reproducibility Notes
Before running the experiments, please verify:
1. model paths;
2. dataset paths;
3. retrieval corpus configuration;
4. RankCoT model checkpoints;
5. large language model API configuration;
6. fine-tuned MacBERT checkpoint;
7. dependencies listed in requirements.txt.
Runtime may vary across different hardware and software environments.
Citation
If this repository is useful for your research, please cite the corresponding paper.
The complete citation information will be added after publication.
## License
Third-party components included or referenced in this repository remain subject to their original licenses.
The use of RankCoT-related code should follow the license terms of the official RankCoT repository.
