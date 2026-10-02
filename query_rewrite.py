
import os
import json
from tqdm import tqdm
from openai import OpenAI
from hybrid_search5 import HybridRetriever


# ============================================================
# 1. GPT API 配置
# ============================================================

API_KEY = os.getenv("OPENAI_API_KEY", "")
API_URL = os.getenv(
    "OPENAI_BASE_URL",
    "https://api.openai.com/v1"
)

MODEL_NAME = "gpt-4o"

client = OpenAI(
    api_key=API_KEY,
    base_url=API_URL
)


# ============================================================
# 2. 文件路径
# ============================================================

# query_diagnosis.py 的输出文件
INPUT_FILE = (
    "/root/autodl-tmp/"
    "low_confidence_with_diagnosis.jsonl"
)

# 查询重写和二次检索后的结果
OUTPUT_FILE = (
    "/root/autodl-tmp/"
    "low_confidence_rewrite_retrieved.jsonl"
)

# 二次检索文档数量
TOP_K = 5


# ============================================================
# 3. 查询缺陷类别与重写策略映射
# ============================================================

DEFECT_STRATEGY = {

    "semantic_ambiguity": {
        "label_cn": "语义歧义",
        "strategy": "语义消歧",
        "instruction": (
            "识别并消除查询中的模糊指代、歧义表达或不明确对象。"
            "在不改变原始问题核心意图的前提下，使查询对象和语义关系更加明确。"
            "不得添加原查询中无法合理确定的新事实。"
        )
    },

    "semantic_missing": {
        "label_cn": "语义缺失",
        "strategy": "信息补全",
        "instruction": (
            "对查询中明显缺失但影响检索理解的语义成分进行最小必要补全。"
            "仅补充能够由原始查询合理推断的表达成分，"
            "不得虚构新的实体、时间、数值或事实条件。"
        )
    },

    "term_nonstandard": {
        "label_cn": "术语表达不规范",
        "strategy": "术语规范化",
        "instruction": (
            "识别查询中的口语化、缩写、不规范或非标准专业术语，"
            "将其改写为规范、统一的专业表达。"
            "不得改变原查询的事实范围和用户意图。"
        )
    },

    "no_obvious_defect": {
        "label_cn": "无明显表达缺陷",
        "strategy": "保守扩展",
        "instruction": (
            "原始查询不存在明显表达缺陷，仅进行保守的检索友好化调整。"
            "可以规范语序或补充同义表达，但不得改变问题语义范围，"
            "不得引入新的事实条件。"
        )
    }
}


# ============================================================
# 4. 构建类别引导查询重写提示词
# ============================================================

def build_rewrite_prompt(
    question: str,
    defect_label: str
) -> str:

    strategy_info = DEFECT_STRATEGY.get(
        defect_label,
        DEFECT_STRATEGY["no_obvious_defect"]
    )

    defect_cn = strategy_info["label_cn"]
    strategy = strategy_info["strategy"]
    instruction = strategy_info["instruction"]

    return f"""
你是一个检索增强问答系统中的查询优化模块。

你的任务是根据查询表达缺陷类别，对原始查询进行定向重写，
生成更适合文档检索的优化查询。

原始查询：
{question}

查询表达缺陷类别：
{defect_cn}

对应重写策略：
{strategy}

具体重写要求：
{instruction}

通用约束：
1. 保持原始查询的核心用户意图不变；
2. 不回答原始问题；
3. 不添加无法从原始查询合理确定的新事实；
4. 不虚构实体、时间、地点、数值或工程参数；
5. 不进行无关扩写；
6. 优先采用最小必要修改；
7. 输出结果应能够直接作为检索查询使用；
8. 如果当前类别下不存在可靠的修改空间，可以保留原始查询。

只输出优化后的查询，不输出解释、标签或其他内容。
"""


# ============================================================
# 5. 调用大语言模型进行类别引导查询重写
# ============================================================

def rewrite_query(
    question: str,
    defect_label: str
) -> str:

    prompt = build_rewrite_prompt(
        question,
        defect_label
    )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    rewritten_query = (
        response
        .choices[0]
        .message
        .content
        .strip()
    )

    if not rewritten_query:
        rewritten_query = question

    return rewritten_query


# ============================================================
# 6. 获取原始查询
# ============================================================

def get_original_query(item):

    query = (
        item.get("query")
        or item.get("original_query")
        or item.get("question")
        or ""
    )

    return query.strip()


# ============================================================
# 7. 获取 MacBERT 查询诊断结果
# ============================================================

def get_diagnosis(item):

    diagnosis = item.get(
        "query_diagnosis",
        {}
    )

    predicted_label = diagnosis.get(
        "predicted_label",
        "no_obvious_defect"
    )

    if predicted_label not in DEFECT_STRATEGY:
        predicted_label = "no_obvious_defect"

    return predicted_label, diagnosis


# ============================================================
# 8. 检查是否属于低置信度样本
# ============================================================

def is_low_confidence(item):

    confidence_info = item.get(
        "confidence_evaluation",
        {}
    )

    epsilon = confidence_info.get(
        "epsilon"
    )

    threshold = confidence_info.get(
        "threshold",
        8.0
    )

    decision = confidence_info.get(
        "decision"
    )

    if decision == "low":
        return True

    if decision == "high":
        return False

    if epsilon is None:
        return True

    try:
        return float(epsilon) < float(threshold)

    except Exception:
        return True


# ============================================================
# 9. 二次检索
# ============================================================

def second_retrieval(
    retriever,
    rewritten_query,
    top_k=5
):

    candidates = retriever.search(
        rewritten_query,
        top_k=top_k
    )

    passages = []

    for rank, candidate in enumerate(
        candidates[:top_k],
        start=1
    ):

        if isinstance(candidate, str):

            text = candidate
            score = None

        elif isinstance(candidate, dict):

            text = (
                candidate.get("text")
                or candidate.get("segment")
                or candidate.get("content")
                or candidate.get("passage")
                or ""
            )

            score = (
                candidate.get("score")
                or candidate.get("similarity")
                or candidate.get("retrieval_score")
            )

        else:

            text = str(candidate)
            score = None

        if not text:
            continue

        passage_item = {
            "rank": rank,
            "segment": text
        }

        if score is not None:

            try:
                passage_item["score"] = float(score)
            except Exception:
                passage_item["score"] = score

        passages.append(
            passage_item
        )

    return passages


# ============================================================
# 10. 单条样本查询优化与二次检索
# ============================================================

def process_one(
    item,
    retriever
):

    original_query = get_original_query(
        item
    )

    if not original_query:
        return None

    if not is_low_confidence(item):
        return None

    predicted_label, diagnosis = (
        get_diagnosis(item)
    )

    strategy_info = DEFECT_STRATEGY[
        predicted_label
    ]

    try:

        rewritten_query = rewrite_query(
            original_query,
            predicted_label
        )

        rewrite_status = "success"

    except Exception as e:

        rewritten_query = original_query
        rewrite_status = "failed"

        rewrite_error = str(e)

    try:

        passages = second_retrieval(
            retriever,
            rewritten_query,
            TOP_K
        )

        retrieval_status = "success"

    except Exception as e:

        passages = []
        retrieval_status = "failed"
        retrieval_error = str(e)

    output_item = dict(item)

    output_item["original_query"] = (
        original_query
    )

    output_item["rewritten_query"] = (
        rewritten_query
    )

    output_item["query_optimization"] = {

        "defect_label":
            predicted_label,

        "defect_label_cn":
            strategy_info["label_cn"],

        "rewrite_strategy":
            strategy_info["strategy"],

        "rewrite_status":
            rewrite_status
    }

    if rewrite_status == "failed":

        output_item[
            "query_optimization"
        ]["rewrite_error"] = rewrite_error

    output_item["second_retrieval"] = {

        "retrieval_query":
            rewritten_query,

        "top_k":
            TOP_K,

        "status":
            retrieval_status,

        "documents":
            passages
    }

    if retrieval_status == "failed":

        output_item[
            "second_retrieval"
        ]["retrieval_error"] = (
            retrieval_error
        )

    return output_item


# ============================================================
# 11. 主流程
# ============================================================

def main():

    if not API_KEY:

        raise ValueError(
            "OPENAI_API_KEY environment variable is not set."
        )

    retriever = HybridRetriever()

    data = []

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            data.append(
                json.loads(line)
            )

    processed_count = 0
    skipped_count = 0

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f_out:

        for item in tqdm(
            data,
            desc="Query rewrite and second retrieval"
        ):

            result = process_one(
                item,
                retriever
            )

            if result is None:

                skipped_count += 1
                continue

            json.dump(
                result,
                f_out,
                ensure_ascii=False
            )

            f_out.write("\n")

            processed_count += 1

    print()
    print("Query optimization completed.")
    print(
        f"Processed samples: "
        f"{processed_count}"
    )

    print(
        f"Skipped samples: "
        f"{skipped_count}"
    )

    print(
        f"Output file: "
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
