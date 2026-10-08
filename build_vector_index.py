import sys
sys.stdout.reconfigure(encoding='utf-8')

# ================== 【关键修复】强制使用 CPU ==================
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # 隐藏所有 GPU，强制 CPU
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

print("=" * 70)
print("构建知识图谱向量索引（CPU 模式）")
print("=" * 70)

# ================== 1. 加载图谱 ==================
print("\n[1/4] 正在加载图谱...")
with open('bank_kg_full.pkl', 'rb') as f:
    G = pickle.load(f)
print(f"  ✅ 图谱加载完成: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")

# ================== 2. 加载嵌入模型 ==================
print("\n[2/4] 正在加载 BGE-M3 嵌入模型（强制 CPU）...")
model = SentenceTransformer('BAAI/bge-m3', device='cpu')  # 强制 CPU
print("  ✅ 模型加载完成")

# ================== 3. 准备节点文本 ==================
print("\n[3/4] 正在准备节点文本...")
node_ids = []
node_texts = []
node_types = []

for node_id, data in G.nodes(data=True):
    node_type = data.get('node_type', 'Unknown')
    
    if node_type == 'Customer':
        text = (f"客户姓名: {data.get('name', '')}, "
                f"年龄: {data.get('age', '')}, "
                f"职业: {data.get('occupation', '')}, "
                f"风险偏好: {data.get('risk_tolerance', '')}, "
                f"闲钱: {data.get('idle_cash', 0)}")
    elif node_type == 'FundProduct':
        text = (f"产品名称: {data.get('name', '')}, "
                f"风险等级: {data.get('risk_level', '')}")
    elif node_type == 'Event':
        text = (f"事件标题: {data.get('title', '')}, "
                f"事件类型: {data.get('event_type', '')}, "
                f"情感得分: {data.get('sentiment_score', 0)}")
    elif node_type == 'MarketScope':
        text = f"市场范围: {node_id}"
    elif node_type == 'BehaviorEvent':
        text = (f"行为事件: {data.get('behavior_type', '')} "
                f"@ {data.get('timestamp', '')}")
    elif node_type == 'FundEvent':
        text = (f"资金事件: {data.get('event_type', '')}, "
                f"金额: {data.get('amount', 0)}")
    else:
        text = f"{node_type}: {node_id}"
    
    node_ids.append(node_id)
    node_texts.append(text)
    node_types.append(node_type)

print(f"  ✅ 准备了 {len(node_texts)} 个节点的文本")

from collections import Counter
type_counter = Counter(node_types)
print("\n  节点类型分布:")
for t, c in type_counter.most_common():
    print(f"    {t}: {c}")

# ================== 4. 分批生成向量（断点续传）==================
print("\n[4/4] 正在生成节点向量（分批处理 + 断点续传）...")

BATCH_SIZE = 2000
PROGRESS_FILE = 'vector_progress.pkl'

# 断点续传：加载已有进度
if os.path.exists(PROGRESS_FILE):
    with open(PROGRESS_FILE, 'rb') as f:
        progress = pickle.load(f)
    embeddings_list = progress['embeddings_list']
    start_batch = progress['next_batch']
    print(f"  📌 检测到已有进度，从第 {start_batch + 1} 批继续")
else:
    embeddings_list = []
    start_batch = 0

num_batches = (len(node_texts) - 1) // BATCH_SIZE + 1

for i in range(start_batch, num_batches):
    start_idx = i * BATCH_SIZE
    end_idx = min((i + 1) * BATCH_SIZE, len(node_texts))
    batch_texts = node_texts[start_idx:end_idx]
    
    print(f"\n  处理第 {i+1}/{num_batches} 批 ({start_idx}-{end_idx}, 共 {len(batch_texts)} 条)...")
    
    batch_emb = model.encode(
        batch_texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
        device='cpu'  # 强制 CPU
    )
    embeddings_list.append(batch_emb)
    
    # 每批保存进度
    with open(PROGRESS_FILE, 'wb') as f:
        pickle.dump({
            'embeddings_list': embeddings_list,
            'next_batch': i + 1
        }, f)
    print(f"  ✅ 第 {i+1} 批完成，进度已保存")

# 合并所有批次的向量
embeddings = np.vstack(embeddings_list)
print(f"\n  ✅ 所有向量生成完成: {embeddings.shape}")

# ================== 5. 构建 FAISS 索引 ==================
print("\n正在构建 FAISS 索引...")
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings.astype('float32'))
print(f"  ✅ FAISS 索引构建完成，共 {index.ntotal} 个向量")

# ================== 6. 保存 ==================
print("\n正在保存...")
faiss.write_index(index, 'kg_vector_index.faiss')

with open('kg_vector_metadata.pkl', 'wb') as f:
    pickle.dump({
        'node_ids': node_ids,
        'node_texts': node_texts,
        'node_types': node_types,
    }, f)

# 删除进度文件（已完成）
if os.path.exists(PROGRESS_FILE):
    os.remove(PROGRESS_FILE)

print("  ✅ 索引已保存到 kg_vector_index.faiss")
print("  ✅ 元数据已保存到 kg_vector_metadata.pkl")
print(f"\n文件大小:")
print(f"  kg_vector_index.faiss: {os.path.getsize('kg_vector_index.faiss') / 1024 / 1024:.1f} MB")
print(f"  kg_vector_metadata.pkl: {os.path.getsize('kg_vector_metadata.pkl') / 1024 / 1024:.1f} MB")
print("\n✅ 全部完成！")