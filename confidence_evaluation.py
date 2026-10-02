import os
import re
import json
import numpy as np
from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

from FlagEmbedding import BGEM3FlagModel


# ============================================================
# 1. 基本配置
# ============================================================

API_KEY = os.getenv("OPENAI_API_KEY", "")
API_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = "gpt-4o"

client = OpenAI(
    api_key=API_KEY,
    base_url=API_URL
)

# 输入问答文件
INPUT_FILE = "/root/autodl-tmp/nq_queryCoT_to_answer.jsonl"

# 钻完井领域规则库
RULE_FILE = "/root/autodl-tmp/drilling_completion_rules.jsonl"

# 输出文件
HIGH_FILE = "high_confidence_multidim.jsonl"
LOW_FILE = "low_confidence_multidim.jsonl"

# 阈值
TAU = 8.0

# 权重自适应强度
ETA = 0.10

# 规则召回数量
RULE_TOP_K = 5

# 并发数
MAX_WORKERS = 8

# 测试数量
# None 表示全部运行
MAX_SAMPLES = None


# ============================================================
# 2. 加载 BGE-M3
# ============================================================

print("Loading BGE-M3...")

bge_model = BGEM3FlagModel(
    "BAAI/bge-m3",
    use_fp16=True
)


# ============================================================
# 3. 读取领域规则库
# ============================================================

def load_rules(rule_file):
    """
    规则文件建议采用 JSONL 格式。

    每行示例：
    {
        "id": "R001",
        "category": "领域术语规范化",
        "rule": "流压和井底流动压力统一规范为井底流压",
        "source": "某钻井工程手册",
        "page": "P35"
    }
    """

    rules = []

    with open(rule_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            item = json.loads(line)

            rule_text = item.get("rule", "").strip()

            if rule_text:
                rules.append(item)

    return rules


RULES = load_rules(RULE_FILE)

RULE_TEXTS = [
    item["rule"]
    for item in RULES
]


# ============================================================
# 4. 预编码领域规则
# ============================================================

print(f"Encoding {len(RULE_TEXTS)} domain rules...")

rule_embeddings = bge_model.encode(
    RULE_TEXTS,
    batch_size=32,
    max_length=512
)["dense_vecs"]

rule_embeddings = np.asarray(rule_embeddings)

# 向量归一化
rule_embeddings = rule_embeddings / (
    np.linalg.norm(rule_embeddings, axis=1, keepdims=True) + 1e-12
)


# ============================================================
# 5. 规则检索
#    R_Q = TopKr[Retriever_R(Q ⊕ A, R)]
# ============================================================

def retrieve_rules(question, answer, top_k=5):
    """
    根据 Q 和 A 从完整领域规则库 R 中召回规则子集 R_Q。
    """

    query_text = f"{question}\n{answer}"

    query_embedding = bge_model.encode(
        [query_text],
        batch_size=1,
        max_length=512
    )["dense_vecs"][0]

    query_embedding = np.asarray(query_embedding)

    query_embedding = query_embedding / (
        np.linalg.norm(query_embedding) + 1e-12
    )

    scores = np.dot(
        rule_embeddings,
        query_embedding
    )

    top_k = min(top_k, len(RULES))

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for idx in top_indices:

        rule_item = RULES[idx].copy()

        rule_item["retrieval_score"] = float(scores[idx])

        results.append(rule_item)

    return results


# ============================================================
# 6. 统一处理检索文档
# ============================================================

def normalize_documents(item):
    """
    兼容不同字段名称。
    优先顺序：
    passages
    documents
    retrieved_docs
    context
    """

    docs = (
        item.get("passages")
        or item.get("documents")
        or item.get("retrieved_docs")
        or item.get("context")
        or []
    )

    if isinstance(docs, str):
        return docs

    if isinstance(docs, list):

        text_list = []

        for i, doc in enumerate(docs):

            if isinstance(doc, str):

                text_list.append(
                    f"[Document {i + 1}]\n{doc}"
                )

            elif isinstance(doc, dict):

                text = (
                    doc.get("text")
                    or doc.get("content")
                    or doc.get("passage")
                    or ""
                )

                text_list.append(
                    f"[Document {i + 1}]\n{text}"
                )

        return "\n\n".join(text_list)

    return str(docs)


# ============================================================
# 7. JSON 提取
# ============================================================

def extract_json(text):
    """
    从 GPT 返回内容中提取 JSON。
    """

    if not text:
        return None

    text = text.strip()

    text = re.sub(
        r"^```json\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"\s*```$",
        "",
        text
    )

    try:
        return json.loads(text)

    except Exception:
        pass

    match = re.search(
        r"\{.*\}",
        text,
        flags=re.DOTALL
    )

    if match:

        try:
            return json.loads(match.group())

        except Exception:
            return None

    return None


# ============================================================
# 8. 分数约束
# ============================================================

def clamp_score(value, low=1.0, high=10.0):

    try:
        value = float(value)

    except Exception:
        value = 5.0

    return max(
        low,
        min(high, value)
    )


# ============================================================
# 9. S1：问题一致性
# ============================================================

def evaluate_s1(question, answer, documents):
    """
    S1 = Judge_M(Q, A, D, I1)

    问题一致性：
    衡量答案是否准确响应原始查询意图。
    """

    prompt = f"""
你是问答质量评估器。

请评价生成答案与用户原始问题之间的一致程度。

评价重点：
1. 答案是否直接回应问题；
2. 是否保持原始查询的核心语义和用户意图；
3. 是否存在答非所问、语义偏移或无关扩展；
4. 检索证据是否能够帮助判断答案与问题之间的一致关系。

评分范围为1到10分：
1表示严重偏离问题；
10表示与问题高度一致。

用户问题：
{question}

模型答案：
{answer}

检索证据：
{documents}

只输出JSON，不输出其他内容。

格式：
{{
    "score": 8.5,
    "reason": "简要评价理由"
}}
"""

    return call_score_model(prompt)


# ============================================================
# 10. S2：证据支撑性
# ============================================================

def evaluate_s2(question, answer, documents):
    """
    S2 = Judge_M(Q, A, D, I2)

    证据支撑性：
    判断检索证据能否支持生成答案。
    """

    prompt = f"""
你是检索增强问答系统中的证据支撑性评估器。

请评价检索证据对模型答案的支撑程度。

评价重点：
1. 检索证据是否与用户问题相关；
2. 模型答案中的主要事实是否能够从检索证据中得到支持；
3. 是否存在证据缺失、证据冲突或证据不足；
4. 不允许仅根据模型自身知识判断答案正确性，应重点依据给定检索证据。

评分范围为1到10分：
1表示答案基本没有证据支撑；
10表示答案中的主要事实均得到充分证据支持。

用户问题：
{question}

模型答案：
{answer}

检索证据：
{documents}

只输出JSON，不输出其他内容。

格式：
{{
    "score": 8.5,
    "reason": "简要评价理由"
}}
"""

    return call_score_model(prompt)


# ============================================================
# 11. S3：事实与逻辑一致性
# ============================================================

def evaluate_s3(
    question,
    answer,
    documents,
    retrieved_rules
):
    """
    S3 = Judge_M(Q, A, D, R_Q, I3)

    事实与逻辑一致性：
    结合检索证据和钻完井领域规则进行判断。
    """

    rule_text = []

    for i, rule in enumerate(retrieved_rules):

        category = rule.get(
            "category",
            "Unknown"
        )

        text = rule.get(
            "rule",
            ""
        )

        source = rule.get(
            "source",
            ""
        )

        rule_text.append(
            f"[Rule {i + 1}]\n"
            f"Category: {category}\n"
            f"Rule: {text}\n"
            f"Source: {source}"
        )

    rule_text = "\n\n".join(rule_text)

    prompt = f"""
你是石油钻完井领域问答质量评估器。

请结合检索证据和钻完井领域规则，评价模型答案的事实与逻辑一致性。

评价重点：
1. 答案中的事实陈述是否与检索证据一致；
2. 专业术语是否符合钻完井领域规范；
3. 实体、井、设备及工程参数之间的关系是否合理；
4. 是否违反给定领域规则；
5. 工程处置建议、参数关系和复杂工况判断是否存在明显逻辑冲突；
6. 若领域规则与当前问题无直接关系，不应强行降低评分。

评分范围为1到10分：
1表示存在严重事实错误或领域逻辑冲突；
10表示事实、证据和领域规则高度一致。

用户问题：
{question}

模型答案：
{answer}

检索证据：
{documents}

相关领域规则：
{rule_text}

只输出JSON，不输出其他内容。

格式：
{{
    "score": 8.5,
    "reason": "简要评价理由"
}}
"""

    return call_score_model(prompt)


# ============================================================
# 12. 通用评分 API
# ============================================================

def call_score_model(prompt):

    try:

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

        raw = response.choices[0].message.content

        result = extract_json(raw)

        if result is None:

            return {
                "score": 5.0,
                "reason": "Failed to parse model output."
            }

        score = clamp_score(
            result.get("score", 5.0)
        )

        return {
            "score": score,
            "reason": result.get(
                "reason",
                ""
            )
        }

    except Exception as e:

        return {
            "score": 5.0,
            "reason": f"API error: {str(e)}"
        }


# ============================================================
# 13. 查询类型判别
#
# P(Q) = TypeJudge_M(Q, I_t)
#      = (p1, p2, p3)
# ============================================================

def classify_query_type(question):

    prompt = f"""
你是查询类型判别器。

请判断下面用户问题在以下三个查询类型上的归属程度。

类型1：语义对齐主导型
主要依赖问题语义理解、用户意图识别、概念解释和语义一致性。

类型2：证据支撑主导型
主要依赖检索证据、事实查找、实体信息和外部知识支撑。

类型3：领域推理主导型
主要涉及专业知识推理、工程逻辑、参数关系、复杂工况判断或处置建议。

请输出三个归属度：
p1、p2、p3。

要求：
1. 每个值位于0到1之间；
2. 三个值之和必须等于1；
3. 可以同时属于多个类型；
4. 不要只进行硬分类。

用户问题：
{question}

只输出JSON。

格式：
{{
    "p1": 0.3,
    "p2": 0.4,
    "p3": 0.3
}}
"""

    try:

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

        raw = response.choices[0].message.content

        result = extract_json(raw)

        if result is None:
            return 1 / 3, 1 / 3, 1 / 3

        p1 = max(
            0.0,
            float(result.get("p1", 0))
        )

        p2 = max(
            0.0,
            float(result.get("p2", 0))
        )

        p3 = max(
            0.0,
            float(result.get("p3", 0))
        )

        total = p1 + p2 + p3

        if total <= 0:
            return 1 / 3, 1 / 3, 1 / 3

        # 强制归一化，保证 p1+p2+p3=1
        p1 /= total
        p2 /= total
        p3 /= total

        return p1, p2, p3

    except Exception:

        return 1 / 3, 1 / 3, 1 / 3


# ============================================================
# 14. 动态权重
#
# wi(Q) = (1-eta)/3 + eta*pi(Q)
#
# eta = 0.1:
# wi(Q) = 0.3 + 0.1*pi(Q)
# ============================================================

def compute_dynamic_weights(
    p1,
    p2,
    p3,
    eta=0.10
):

    base = (1.0 - eta) / 3.0

    w1 = base + eta * p1
    w2 = base + eta * p2
    w3 = base + eta * p3

    return w1, w2, w3


# ============================================================
# 15. 单条样本多维置信度计算
# ============================================================

def evaluate_one(item):

    question = item.get(
        "query",
        ""
    )

    answer = (
        item.get("model_answer")
        or item.get("answer")
        or ""
    )

    documents = normalize_documents(item)

    # --------------------------------------------------------
    # Step 1：查询类型判别
    # --------------------------------------------------------

    p1, p2, p3 = classify_query_type(
        question
    )

    # --------------------------------------------------------
    # Step 2：计算动态权重
    # --------------------------------------------------------

    w1, w2, w3 = compute_dynamic_weights(
        p1,
        p2,
        p3,
        ETA
    )

    # --------------------------------------------------------
    # Step 3：问题一致性 S1
    # --------------------------------------------------------

    s1_result = evaluate_s1(
        question,
        answer,
        documents
    )

    # --------------------------------------------------------
    # Step 4：证据支撑性 S2
    # --------------------------------------------------------

    s2_result = evaluate_s2(
        question,
        answer,
        documents
    )

    # --------------------------------------------------------
    # Step 5：领域规则检索 R_Q
    # --------------------------------------------------------

    retrieved_rules = retrieve_rules(
        question,
        answer,
        RULE_TOP_K
    )

    # --------------------------------------------------------
    # Step 6：事实与逻辑一致性 S3
    # --------------------------------------------------------

    s3_result = evaluate_s3(
        question,
        answer,
        documents,
        retrieved_rules
    )

    S1 = s1_result["score"]
    S2 = s2_result["score"]
    S3 = s3_result["score"]

    # --------------------------------------------------------
    # Step 7：综合置信度
    #
    # epsilon = w1*S1 + w2*S2 + w3*S3
    # --------------------------------------------------------

    epsilon = (
        w1 * S1
        + w2 * S2
        + w3 * S3
    )

    # --------------------------------------------------------
    # Step 8：阈值判断
    # --------------------------------------------------------

    if epsilon >= TAU:
        decision = "high"
    else:
        decision = "low"

    # --------------------------------------------------------
    # Step 9：保存详细评价结果
    # --------------------------------------------------------

    item["confidence_evaluation"] = {

        "query_type_probability": {
            "p1_semantic_alignment": round(p1, 6),
            "p2_evidence_support": round(p2, 6),
            "p3_domain_reasoning": round(p3, 6)
        },

        "dynamic_weights": {
            "w1": round(w1, 6),
            "w2": round(w2, 6),
            "w3": round(w3, 6)
        },

        "scores": {
            "S1_question_consistency": S1,
            "S2_evidence_support": S2,
            "S3_fact_logic_consistency": S3
        },

        "score_reasons": {
            "S1": s1_result["reason"],
            "S2": s2_result["reason"],
            "S3": s3_result["reason"]
        },

        "epsilon": round(
            epsilon,
            6
        ),

        "threshold": TAU,

        "decision": decision,

        "retrieved_rules": retrieved_rules
    }

    return item


# ============================================================
# 16. 主流程
# ============================================================

def main():

    if not API_KEY:
        raise ValueError(
            "OPENAI_API_KEY environment variable is not set."
        )

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

    if MAX_SAMPLES is not None:
        data = data[:MAX_SAMPLES]

    high_count = 0
    low_count = 0

    with open(
        HIGH_FILE,
        "w",
        encoding="utf-8"
    ) as f_high, open(
        LOW_FILE,
        "w",
        encoding="utf-8"
    ) as f_low:

        with ThreadPoolExecutor(
            max_workers=MAX_WORKERS
        ) as executor:

            futures = [
                executor.submit(
                    evaluate_one,
                    item
                )
                for item in data
            ]

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Multidimensional confidence evaluation"
            ):

                try:

                    item = future.result()

                except Exception as e:

                    print(
                        f"Sample processing failed: {e}"
                    )

                    continue

                epsilon = item[
                    "confidence_evaluation"
                ]["epsilon"]

                if epsilon >= TAU:

                    json.dump(
                        item,
                        f_high,
                        ensure_ascii=False
                    )

                    f_high.write("\n")

                    high_count += 1

                else:

                    json.dump(
                        item,
                        f_low,
                        ensure_ascii=False
                    )

                    f_low.write("\n")

                    low_count += 1

    print()
    print("Evaluation completed.")
    print(
        f"High confidence (epsilon >= {TAU}): "
        f"{high_count}"
    )

    print(
        f"Low confidence (epsilon < {TAU}): "
        f"{low_count}"
    )

    print()
    print("Output files:")
    print(f"High: {HIGH_FILE}")
    print(f"Low : {LOW_FILE}")


if __name__ == "__main__":
    main()
