import os
import random
import logging
import copy
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler
from transformers import BertTokenizer, BertForMaskedLM

# --- 組態設定 ---
class Config:
    # <--- MOD: 在此設定您想一次執行的所有任務 --- >
    TASKS_TO_TEST = ["SST-2", "QNLI"]

    # 測試參數
    EPSILON_VALUES = [1.0, 2.0, 3.0]
    METHODS_TO_TEST = ["SanText", "SanText+", "SanText+_NER"]

    # SanText+ 和 SanText+_NER 的特定參數
    SANTEXT_PLUS_P = 0.3
    SANTEXT_PLUS_SENSITIVE_WORD_PERCENTAGE = 0.9

    # 模型路徑 (共用)
    MODEL_PATH = "bert-base-uncased"
    
    # <--- MOD: 將路徑改為字典結構，以任務名為鍵 --- >
    # 原始資料路徑
    ORIGINAL_DATA_DIRS = {
        "SST-2": "./data/SST-2/",
        "QNLI": "./data/QNLI/"
    }
    
    # 匿名化資料的基礎路徑
    SANITIZED_BASE_DIRS = {
        "SST-2": {
            "SanText": "./output_SanText/SST-2/",
            "SanText+": "./output_SanText_plus_glove/SST-2/",
            "SanText+_NER": "./output_SanText_plus_glove_NER/SST-2/"
        },
        "QNLI": {
            "SanText": "./output_SanText/QNLI/",
            "SanText+": "./output_SanText_plus_glove/QNLI/",
            "SanText+_NER": "./output_SanText_plus_glove_NER/QNLI/"
        }
    }
    
    # 執行參數
    MAX_SEQ_LENGTH = 128
    BATCH_SIZE = 64
    SEED = 42

# --- 輔助函式 ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# <--- MOD: 建立一個更通用的資料讀取函式 --- >
def load_glue_data(task_name, file_path, tokenizer):
    """從 .tsv 檔案載入不同 GLUE 任務的資料並進行 tokenize"""
    docs = []
    if not os.path.exists(file_path):
        logging.warning(f"檔案不存在: {file_path}")
        return None
        
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            next(f) # 跳過標頭
        except StopIteration:
            return [] # 空檔案

        for line in f:
            parts = line.strip().split("\t")
            if task_name == "SST-2":
                if len(parts) >= 1:
                    docs.append(tokenizer.tokenize(parts[0]))
            elif task_name == "QNLI":
                if len(parts) >= 3:
                    # QNLI 需要處理問題和句子兩個部分
                    docs.append(tokenizer.tokenize(parts[1])) # Question
                    docs.append(tokenizer.tokenize(parts[2])) # Sentence
    return docs

# --- 主要驗證邏輯 (不變) ---
def verify_privacy_from_files(model, tokenizer, original_docs, sanitized_docs, device):
    # ... 此函式內部邏輯完全不變，故省略以節省篇幅 ...
    inference_samples, ground_truth_labels = [], []
    doc_count = min(len(original_docs), len(sanitized_docs))
    for i in tqdm(range(doc_count), desc="準備推斷樣本"):
        original_doc, sanitized_doc = original_docs[i], sanitized_docs[i]
        if len(sanitized_doc) != len(original_doc): continue
        for j in range(len(sanitized_doc)):
            masked_doc = copy.deepcopy(sanitized_doc)
            masked_doc[j] = "[MASK]"
            text = tokenizer.convert_tokens_to_string(masked_doc)
            inputs = tokenizer.encode_plus(text, padding="max_length", max_length=Config.MAX_SEQ_LENGTH, truncation=True)
            try:
                mask_token_index = inputs['input_ids'].index(tokenizer.mask_token_id)
                inference_samples.append(inputs)
                original_token_id = tokenizer.convert_tokens_to_ids(original_doc[j])
                ground_truth_labels.append((mask_token_index, original_token_id))
            except (ValueError, KeyError):
                continue
    if not inference_samples: return 0.0
    dataset = TensorDataset(torch.tensor([s['input_ids'] for s in inference_samples]), torch.tensor([s['attention_mask'] for s in inference_samples]), torch.tensor([s['token_type_ids'] for s in inference_samples]))
    dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=Config.BATCH_SIZE)
    model.eval()
    intersect_num, total_num = 0, 0
    for i, batch in enumerate(tqdm(dataloader, desc="執行推斷")):
        batch = tuple(t.to(device) for t in batch)
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2]}
        with torch.no_grad():
            predictions = model(**inputs).logits
        for j in range(len(batch[0])):
            global_idx = i * Config.BATCH_SIZE + j
            if global_idx >= len(ground_truth_labels): continue
            mask_idx, label_id = ground_truth_labels[global_idx]
            predicted_token_id = torch.argmax(predictions[j, mask_idx, :]).item()
            if predicted_token_id == label_id: intersect_num += 1
            total_num += 1
    accuracy = (intersect_num / total_num) if total_num > 0 else 0.0
    logging.info(f"推斷完成: 總共 {total_num} 個 token, 猜對 {intersect_num} 個。")
    return accuracy

# --- 主執行流程 ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"使用裝置: {device}")

    logging.info(f"載入共用模型: {Config.MODEL_PATH}")
    tokenizer = BertTokenizer.from_pretrained(Config.MODEL_PATH)
    model = BertForMaskedLM.from_pretrained(Config.MODEL_PATH).to(device)

    all_results = []

    # <--- MOD: 新增最外層的任務迴圈 --- >
    for task_name in Config.TASKS_TO_TEST:
        logging.info(f"================== 開始處理任務: {task_name} ==================")
        
        # 根據任務名獲取對應的原始資料路徑
        original_data_dir = Config.ORIGINAL_DATA_DIRS.get(task_name)
        if not original_data_dir:
            logging.error(f"在 Config 中找不到任務 '{task_name}' 的原始資料路徑設定，跳過。")
            continue

        original_dev_file = os.path.join(original_data_dir, 'dev.tsv')
        logging.info(f"載入任務 '{task_name}' 的原始開發集: {original_dev_file}")
        
        # 使用新的通用函式載入資料
        original_docs = load_glue_data(task_name, original_dev_file, tokenizer)
        if original_docs is None:
            logging.error(f"無法載入原始開發集 {original_dev_file}，跳過任務 {task_name}。")
            continue

        # 根據任務名獲取對應的匿名化資料基礎路徑字典
        sanitized_base_dirs_for_task = Config.SANITIZED_BASE_DIRS.get(task_name)
        if not sanitized_base_dirs_for_task:
            logging.error(f"在 Config 中找不到任務 '{task_name}' 的匿名化資料路徑設定，跳過。")
            continue

        for epsilon in Config.EPSILON_VALUES:
            logging.info(f"====== 測試 Epsilon = {epsilon} (任務: {task_name}) ======")
            for method in Config.METHODS_TO_TEST:
                logging.info(f"--- 測試方法: {method} ---")
                
                base_dir = sanitized_base_dirs_for_task.get(method)
                if not base_dir:
                    logging.warning(f"找不到方法 '{method}' (任務: {task_name}) 的基礎路徑設定，跳過測試。")
                    continue

                if method == "SanText":
                    sanitized_dir = Path(base_dir) / f"eps_{epsilon:.2f}"
                elif method == "SanText+":
                    sanitized_dir = Path(base_dir) / f"eps_{epsilon:.2f}" / f"sword_{Config.SANTEXT_PLUS_SENSITIVE_WORD_PERCENTAGE:.2f}_p_{Config.SANTEXT_PLUS_P:.2f}"
                elif method == "SanText+_NER":
                    sanitized_dir = Path(base_dir) / f"eps_{epsilon:.2f}" / f"sword_{Config.SANTEXT_PLUS_SENSITIVE_WORD_PERCENTAGE:.2f}_p_{Config.SANTEXT_PLUS_P:.2f}_NER"
                else:
                    logging.warning(f"未知的測試方法 '{method}'，跳過。")
                    continue
                
                sanitized_dev_file = sanitized_dir / "dev.tsv"
                logging.info(f"嘗試讀取匿名化檔案: {sanitized_dev_file}")

                if not sanitized_dev_file.exists():
                    logging.warning(f"檔案不存在！請先執行對應的 run_SanText.py 產生此檔案。")
                    logging.warning(f"跳過 任務={task_name}, Epsilon={epsilon}, Method={method} 的測試。")
                    continue

                sanitized_docs = load_glue_data(task_name, sanitized_dev_file, tokenizer)
                if sanitized_docs is None:
                    logging.warning(f"無法載入匿名化開發集 {sanitized_dev_file}，跳過測試。")
                    continue
                
                accuracy = verify_privacy_from_files(model, tokenizer, original_docs, sanitized_docs, device)
                
                # <--- MOD: 將任務名稱也存入結果中 --- >
                all_results.append({ "task": task_name, "epsilon": epsilon, "method": method, "inference_accuracy": accuracy })
                logging.info(f"結果: Task={task_name}, Epsilon={epsilon}, Method={method}, 推斷準確率={accuracy:.4f}")

    # <--- MOD: 修改報告格式以包含任務名稱 --- >
    print("\n\n===== 隱私性驗證總結報告 (多任務) =====")
    print(f"{'Task':<10} | {'Epsilon':<10} | {'Method':<15} | {'Inference Accuracy (Privacy Leakage)':<40}")
    print("-" * 85)
    for res in all_results:
        print(f"{res['task']:<10} | {res['epsilon']:<10.2f} | {res['method']:<15} | {res['inference_accuracy']:.4f}")
    print("\n報告解讀：推斷準確率越低，代表攻擊者越難猜出原始單詞，隱私保護效果越好。")