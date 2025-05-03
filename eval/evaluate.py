# eval/evaluate.py

import pandas as pd
import requests
import time
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer

CSV_PATH = "data/viquad-mini.csv"
API_URL = "http://localhost:8000/chat"

# Hàm tiền xử lý text đơn giản
def normalize(text):
    return text.lower().strip().replace("\n", " ")

# Hàm tính F1 đơn giản cho từng câu
def simple_f1(pred, ref):
    pred_tokens = set(normalize(pred).split())
    ref_tokens = set(normalize(ref).split())
    if not pred_tokens or not ref_tokens:
        return 0.0
    overlap = pred_tokens & ref_tokens
    precision = len(overlap) / len(pred_tokens)
    recall = len(overlap) / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

# Đọc dữ liệu CSV
samples = pd.read_csv(CSV_PATH)

f1_scores = []
latencies = []

print("\n🔍 Đang đánh giá mô hình trên 20 câu hỏi...")
for idx, row in samples.iterrows():
    question = row["question"]
    gold_answer = row["answer"]

    start = time.time()
    try:
        response = requests.post(API_URL, json={"prompt": question})
        response.raise_for_status()
        prediction = response.json()["response"]
    except Exception as e:
        print(f"❌ Lỗi câu {idx+1}: {e}")
        prediction = ""

    end = time.time()
    latency = end - start

    f1 = simple_f1(prediction, gold_answer)
    f1_scores.append(f1)
    latencies.append(latency)

    print(f"✅ Q{idx+1}: F1={f1:.2f}, Latency={latency:.2f}s")

# Tổng hợp kết quả
avg_f1 = sum(f1_scores) / len(f1_scores)
avg_latency = sum(latencies) / len(latencies)

print("\n📊 Kết quả đánh giá:")
print(f"🎯 F1 trung bình: {avg_f1:.3f}")
print(f"⚡ Độ trễ trung bình: {avg_latency:.2f} giây")
