import sys
sys.stdout.reconfigure(encoding='utf-8')

import torch
import torch.nn as nn
import numpy as np
import pickle
import json

# ================== 1. 加载数据和模型 ==================
print("正在加载数据...")
with open('gnn_data.pkl', 'rb') as f:
    data = pickle.load(f)

customer_embeddings = np.load('customer_embeddings.npy')
product_embeddings = np.load('product_embeddings.npy')

customers = data['customer_ids']
products = data['product_ids']

print(f"客户嵌入: {customer_embeddings.shape}")
print(f"产品嵌入: {product_embeddings.shape}")


# ================== 2. 推荐函数 ==================
def recommend_for_customer(customer_id, top_k=3):
    """
    给定客户ID，返回 Top-K 推荐产品
    """
    if customer_id not in customers:
        return []
    
    c_idx = customers.index(customer_id)
    c_emb = customer_embeddings[c_idx]
    
    # 计算该客户与所有产品的得分
    scores = []
    for i, pid in enumerate(products):
        p_emb = product_embeddings[i]
        score = float(np.dot(c_emb, p_emb))
        scores.append((pid, score))
    
    # 排序
    scores.sort(key=lambda x: x[1], reverse=True)
    
    return scores[:top_k]


# ================== 3. 批量推荐 ==================
print("\n" + "=" * 70)
print("📊 GNN 推荐结果")
print("=" * 70)

# 演示：为前10个客户推荐
recommendations = []
for cid in customers[:10]:
    recs = recommend_for_customer(cid, top_k=3)
    recommendations.append({
        'customer_id': cid,
        'recommendations': [
            {'product_id': pid, 'score': score}
            for pid, score in recs
        ]
    })
    print(f"\n客户 {cid}:")
    for pid, score in recs:
        print(f"  → {pid} (得分: {score:.4f})")


# ================== 4. 对比规则引擎 ==================
print("\n" + "=" * 70)
print("📊 对比：GNN 推荐 vs 规则推荐")
print("=" * 70)

# 加载规则推荐结果
with open('recommendations.json', 'r', encoding='utf-8') as f:
    rule_recs = json.load(f)

print(f"\n规则引擎推荐总数: {sum(len(v) for v in rule_recs.values())}")
print(f"GNN 推荐总数: {len(recommendations) * 3}")

# 保存
with open('gnn_recommendations.json', 'w', encoding='utf-8') as f:
    json.dump(recommendations, f, ensure_ascii=False, indent=2)

print(f"\n✅ GNN 推荐结果已保存到 gnn_recommendations.json")