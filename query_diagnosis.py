
import os
import json
import random
import argparse

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm


# ============================================================
# 1. 基本配置
# ============================================================

MODEL_NAME = "hfl/chinese-macbert-base"

# 训练、验证、测试数据
TRAIN_FILE = "/root/autodl-tmp/query_diagnosis_train.jsonl"
DEV_FILE = "/root/autodl-tmp/query_diagnosis_dev.jsonl"
TEST_FILE = "/root/autodl-tmp/query_diagnosis_test.jsonl"

# 前一阶段多维置信度评估产生的低置信度样本
LOW_CONFIDENCE_FILE = "low_confidence_multidim.jsonl"

# 查询诊断后的输出文件
DIAGNOSIS_OUTPUT_FILE = "low_confidence_with_diagnosis.jsonl"

# 模型保存目录
SAVE_DIR = "./macbert_query_diagnosis"

MAX_LENGTH = 128
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 5
WEIGHT_DECAY = 0.01

RANDOM_SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# 2. 四类查询表达缺陷
#
# 对应论文式(16)-(20)
# ============================================================

LABEL2ID = {
    "semantic_ambiguity": 0,
    "semantic_missing": 1,
    "term_nonstandard": 2,
    "no_obvious_defect": 3
}

ID2LABEL = {
    0: "semantic_ambiguity",
    1: "semantic_missing",
    2: "term_nonstandard",
    3: "no_obvious_defect"
}

LABEL2CN = {
    "semantic_ambiguity": "语义歧义",
    "semantic_missing": "语义缺失",
    "term_nonstandard": "术语表达不规范",
    "no_obvious_defect": "无明显表达缺陷"
}


# ============================================================
# 3. 固定随机种子
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(RANDOM_SEED)


# ============================================================
# 4. 数据读取
#
# 论文：
# Dq = {(q_i, y_i)}_{i=1}^{N_q}
#
# 每条数据建议：
#
# {
#     "query": "流压是多少",
#     "label": "term_nonstandard"
# }
# ============================================================

def load_jsonl(file_path):

    data = []

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            item = json.loads(line)

            query = item.get(
                "query",
                ""
            ).strip()

            label = item.get(
                "label",
                ""
            ).strip()

            if not query:
                continue

            if label not in LABEL2ID:
                continue

            data.append({
                "query": query,
                "label": label,
                "label_id": LABEL2ID[label]
            })

    return data


# ============================================================
# 5. Dataset
#
# 对应论文式(18)
#
# x_i = [CLS] q_i [SEP]
# ============================================================

class QueryDiagnosisDataset(Dataset):

    def __init__(
        self,
        data,
        tokenizer,
        max_length=128
    ):

        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        item = self.data[index]

        query = item["query"]
        label = item["label_id"]

        encoding = self.tokenizer(
            query,
            add_special_tokens=True,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_attention_mask=True,
            return_tensors="pt"
        )

        return {
            "input_ids":
                encoding["input_ids"].squeeze(0),

            "attention_mask":
                encoding["attention_mask"].squeeze(0),

            "labels":
                torch.tensor(
                    label,
                    dtype=torch.long
                )
        }


# ============================================================
# 6. 初始化 MacBERT-base
#
# 对应论文：
# 使用 [CLS] 隐状态作为查询整体语义表示
# 然后通过四分类线性层得到类别概率
# ============================================================

def build_model():

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=4,
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )

    model.to(DEVICE)

    return tokenizer, model


# ============================================================
# 7. 单轮训练
#
# 对应论文式(21)
#
# L = - sum y_ik log p_ik
#
# AutoModelForSequenceClassification
# 内部直接计算 CrossEntropyLoss
# ============================================================

def train_one_epoch(
    model,
    dataloader,
    optimizer,
    scheduler
):

    model.train()

    total_loss = 0.0

    progress_bar = tqdm(
        dataloader,
        desc="Training"
    )

    for batch in progress_bar:

        input_ids = batch[
            "input_ids"
        ].to(DEVICE)

        attention_mask = batch[
            "attention_mask"
        ].to(DEVICE)

        labels = batch[
            "labels"
        ].to(DEVICE)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    avg_loss = (
        total_loss / len(dataloader)
    )

    return avg_loss


# ============================================================
# 8. 模型评价
#
# 输出：
# ACC
# Macro-F1
#
# 对应论文3.3和4.3
# ============================================================

def evaluate(
    model,
    dataloader,
    return_predictions=False
):

    model.eval()

    all_labels = []
    all_predictions = []
    all_probabilities = []

    total_loss = 0.0

    with torch.no_grad():

        for batch in tqdm(
            dataloader,
            desc="Evaluating"
        ):

            input_ids = batch[
                "input_ids"
            ].to(DEVICE)

            attention_mask = batch[
                "attention_mask"
            ].to(DEVICE)

            labels = batch[
                "labels"
            ].to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            logits = outputs.logits

            probabilities = torch.softmax(
                logits,
                dim=-1
            )

            predictions = torch.argmax(
                probabilities,
                dim=-1
            )

            total_loss += loss.item()

            all_labels.extend(
                labels.cpu().numpy().tolist()
            )

            all_predictions.extend(
                predictions.cpu().numpy().tolist()
            )

            all_probabilities.extend(
                probabilities.cpu().numpy().tolist()
            )

    avg_loss = (
        total_loss / len(dataloader)
    )

    acc = accuracy_score(
        all_labels,
        all_predictions
    )

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro"
    )

    result = {
        "loss": avg_loss,
        "accuracy": acc,
        "macro_f1": macro_f1
    }

    if return_predictions:

        result["labels"] = all_labels
        result["predictions"] = all_predictions
        result["probabilities"] = all_probabilities

    return result


# ============================================================
# 9. 全参数监督微调
#
# 对应论文2.3.1(e)
#
# 不冻结 MacBERT 编码器
# encoder + classifier 同时更新
# ============================================================

def train():

    print("Loading training data...")

    train_data = load_jsonl(
        TRAIN_FILE
    )

    dev_data = load_jsonl(
        DEV_FILE
    )

    test_data = load_jsonl(
        TEST_FILE
    )

    print(
        f"Train samples: {len(train_data)}"
    )

    print(
        f"Dev samples: {len(dev_data)}"
    )

    print(
        f"Test samples: {len(test_data)}"
    )

    tokenizer, model = build_model()

    train_dataset = QueryDiagnosisDataset(
        train_data,
        tokenizer,
        MAX_LENGTH
    )

    dev_dataset = QueryDiagnosisDataset(
        dev_data,
        tokenizer,
        MAX_LENGTH
    )

    test_dataset = QueryDiagnosisDataset(
        test_data,
        tokenizer,
        MAX_LENGTH
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # --------------------------------------------------------
    # 全参数更新
    # --------------------------------------------------------

    for parameter in model.parameters():
        parameter.requires_grad = True

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    total_steps = (
        len(train_loader)
        * NUM_EPOCHS
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(
            total_steps * 0.1
        ),
        num_training_steps=total_steps
    )

    best_macro_f1 = -1.0

    os.makedirs(
        SAVE_DIR,
        exist_ok=True
    )

    for epoch in range(
        1,
        NUM_EPOCHS + 1
    ):

        print()
        print(
            f"Epoch {epoch}/{NUM_EPOCHS}"
        )

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler
        )

        dev_result = evaluate(
            model,
            dev_loader
        )

        print(
            f"Train Loss: "
            f"{train_loss:.4f}"
        )

        print(
            f"Dev Loss: "
            f"{dev_result['loss']:.4f}"
        )

        print(
            f"Dev ACC: "
            f"{dev_result['accuracy'] * 100:.2f}%"
        )

        print(
            f"Dev Macro-F1: "
            f"{dev_result['macro_f1'] * 100:.2f}%"
        )

        # ----------------------------------------------------
        # 按验证集 Macro-F1 保存最优模型
        # ----------------------------------------------------

        if (
            dev_result["macro_f1"]
            > best_macro_f1
        ):

            best_macro_f1 = (
                dev_result["macro_f1"]
            )

            model.save_pretrained(
                SAVE_DIR
            )

            tokenizer.save_pretrained(
                SAVE_DIR
            )

            with open(
                os.path.join(
                    SAVE_DIR,
                    "label_mapping.json"
                ),
                "w",
                encoding="utf-8"
            ) as f:

                json.dump(
                    {
                        "label2id": LABEL2ID,
                        "id2label": ID2LABEL,
                        "label2cn": LABEL2CN
                    },
                    f,
                    ensure_ascii=False,
                    indent=2
                )

            print(
                "Best model saved."
            )

    # ========================================================
    # 10. 测试集最终评价
    # ========================================================

    print()
    print("Evaluating best model on test set...")

    best_model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            SAVE_DIR
        )
    )

    best_model.to(
        DEVICE
    )

    test_result = evaluate(
        best_model,
        test_loader,
        return_predictions=True
    )

    print()
    print(
        f"Test ACC: "
        f"{test_result['accuracy'] * 100:.2f}%"
    )

    print(
        f"Test Macro-F1: "
        f"{test_result['macro_f1'] * 100:.2f}%"
    )

    print()

    print(
        classification_report(
            test_result["labels"],
            test_result["predictions"],
            labels=[0, 1, 2, 3],
            target_names=[
                "Semantic Ambiguity",
                "Semantic Missing",
                "Term Nonstandard",
                "No Obvious Defect"
            ],
            digits=4,
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # 保存测试结果
    # --------------------------------------------------------

    metrics = {
        "test_samples":
            len(test_data),

        "accuracy":
            test_result["accuracy"],

        "macro_f1":
            test_result["macro_f1"],

        "random_seed":
            RANDOM_SEED,

        "max_length":
            MAX_LENGTH,

        "batch_size":
            BATCH_SIZE,

        "learning_rate":
            LEARNING_RATE,

        "epochs":
            NUM_EPOCHS
    }

    with open(
        os.path.join(
            SAVE_DIR,
            "test_metrics.json"
        ),
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metrics,
            f,
            ensure_ascii=False,
            indent=2
        )


# ============================================================
# 11. 加载训练后的 MacBERT
# ============================================================

def load_diagnosis_model():

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            SAVE_DIR
        )
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            SAVE_DIR
        )
    )

    model.to(
        DEVICE
    )

    model.eval()

    return tokenizer, model


# ============================================================
# 12. 单条查询缺陷诊断
#
# 对应论文式(19)
#
# p_i = Softmax(W h_[CLS] + b)
#
# 对应论文式(20)
#
# y_hat = argmax_k p(y=k | q)
# ============================================================

def diagnose_query(
    query,
    tokenizer,
    model
):

    encoding = tokenizer(
        query,
        add_special_tokens=True,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    input_ids = (
        encoding["input_ids"]
        .to(DEVICE)
    )

    attention_mask = (
        encoding["attention_mask"]
        .to(DEVICE)
    )

    with torch.no_grad():

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        logits = outputs.logits

        probabilities = torch.softmax(
            logits,
            dim=-1
        )[0]

        predicted_id = int(
            torch.argmax(
                probabilities
            ).item()
        )

    predicted_label = (
        ID2LABEL[predicted_id]
    )

    result = {
        "predicted_class_id":
            predicted_id,

        "predicted_label":
            predicted_label,

        "predicted_label_cn":
            LABEL2CN[
                predicted_label
            ],

        "probabilities": {
            "semantic_ambiguity":
                round(
                    float(
                        probabilities[0]
                    ),
                    6
                ),

            "semantic_missing":
                round(
                    float(
                        probabilities[1]
                    ),
                    6
                ),

            "term_nonstandard":
                round(
                    float(
                        probabilities[2]
                    ),
                    6
                ),

            "no_obvious_defect":
                round(
                    float(
                        probabilities[3]
                    ),
                    6
                )
        }
    }

    return result


# ============================================================
# 13. 与多维置信度评估模块衔接
#
# 输入：
# low_confidence_multidim.jsonl
#
# 该文件由 confidence_evaluation.py 产生：
#
# epsilon < tau
#
# 才进入当前 MacBERT 查询缺陷诊断模块。
# ============================================================

def diagnose_low_confidence_samples():

    print(
        "Loading trained MacBERT model..."
    )

    tokenizer, model = (
        load_diagnosis_model()
    )

    data = []

    with open(
        LOW_CONFIDENCE_FILE,
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

    print(
        f"Low-confidence samples: "
        f"{len(data)}"
    )

    with open(
        DIAGNOSIS_OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as fout:

        for item in tqdm(
            data,
            desc="Query diagnosis"
        ):

            question = item.get(
                "query",
                ""
            )

            # ------------------------------------------------
            # 再检查一次置信度条件
            #
            # 与前一阶段 epsilon < tau 对应
            # ------------------------------------------------

            confidence_info = (
                item.get(
                    "confidence_evaluation",
                    {}
                )
            )

            epsilon = (
                confidence_info.get(
                    "epsilon",
                    None
                )
            )

            threshold = (
                confidence_info.get(
                    "threshold",
                    8.0
                )
            )

            if epsilon is not None:

                if epsilon >= threshold:
                    continue

            diagnosis = diagnose_query(
                question,
                tokenizer,
                model
            )

            item[
                "query_diagnosis"
            ] = diagnosis

            json.dump(
                item,
                fout,
                ensure_ascii=False
            )

            fout.write("\n")

    print()
    print(
        "Query diagnosis completed."
    )

    print(
        f"Output: "
        f"{DIAGNOSIS_OUTPUT_FILE}"
    )


# ============================================================
# 14. 主函数
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=[
            "train",
            "diagnose"
        ]
    )

    args = parser.parse_args()

    if args.mode == "train":

        train()

    elif args.mode == "diagnose":

        diagnose_low_confidence_samples()


if __name__ == "__main__":
    main()
