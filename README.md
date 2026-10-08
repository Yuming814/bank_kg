# 🏦 零售银行 - 事件驱动产品推荐系统

> 基于**知识图谱 + 规则引擎 + GNN + RAG-FLARKO**的智能推荐系统

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![NetworkX](https://img.shields.io/badge/NetworkX-3.0+-green.svg)](https://networkx.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io/)
[![RAG](https://img.shields.io/badge/RAG-FLARKO-purple.svg)](https://github.com/brains-group/FLARKO)

---

## 📖 项目简介

零售银行的产品推荐面临三大挑战：

1. **客户维度缺失**：传统推荐只看"产品-产品"关联，忽略客户个性化需求
2. **事件驱动不足**：市场事件（加息、股市波动）无法自动触发推荐
3. **可解释性差**：黑盒模型无法解释"为什么推荐这个产品"

本项目构建了一个 **"事件-产品-客户"三位一体**的知识图谱，并结合**三套推荐引擎**，实现了从市场事件到客户推荐的完整闭环：

- **规则引擎**：4 条业务规则，可解释、响应快
- **GNN 模型**：自动学习高阶关系，个性化推荐
- **RAG-FLARKO**：对话式智能顾问，推荐理由可追溯

---

## 🏗️ 系统架构
┌─────────────────────────────────────────────────────┐
│ 数据层 (Data Layer) │
│ 客户360°画像 + 产品库 + 购买记录 + 行为日志 + 市场事件 │
└─────────────────────────────────────────────────────┘
↓
┌─────────────────────────────────────────────────────┐
│ 知识图谱层 (Knowledge Graph Layer) │
│ 6 层异构图谱: 产品维度 / 市场范围 / 持仓 / 事件 / 客户 / 行为 │
│ 19,029 节点, 44,431 边 │
└─────────────────────────────────────────────────────┘
↓
┌─────────────────────────────────────────────────────┐
│ 推荐引擎层 (Recommendation Layer) │
│ ① 规则引擎（4条规则） ② GNN 异构图神经网络 │
│ ③ RAG-FLARKO（多阶段检索 + Reranker + LLM） │
└─────────────────────────────────────────────────────┘
↓
┌─────────────────────────────────────────────────────┐
│ 应用层 (Application Layer) │
│ Streamlit 交互式仪表盘 + 智能顾问对话 │
└─────────────────────────────────────────────────────┘

---

## ✨ 核心特性

### 📊 知识图谱建模

- **6 层异构图谱**：产品维度 → 市场范围 → 产品持仓 → 事件 → 客户 → 行为
- **14 类节点**：Customer、FundProduct、Event、MarketScope、AssetClass 等
- **15 种关系**：HOLDS、PREFERS、ABOUT、HAS_EXPOSURE_TO、TRIGGERS 等

### 🎯 三套推荐引擎

| 引擎 | 特点 | 适用场景 |
|------|------|---------|
| **规则引擎** | 4 条业务规则，可解释，响应快 | 快速上线，规则明确 |
| **GNN 模型** | 自动学习高阶关系，个性化 | 复杂场景，智能推荐 |
| **RAG-FLARKO** | 对话式，可解释，LLM 生成 | 客户交互，智能顾问 |

### 🧠 RAG-FLARKO 智能顾问

基于 **多阶段知识图谱检索 + Reranker + LLM** 的对话式投资顾问：

- **多阶段检索**：先从个人知识图谱（PKG）检索客户画像，再从市场知识图谱（MKG）检索相关事件
- **Reranker 二次排序**：用 `BGE-reranker-v2-m3` 精准重排，业务相关性提升 10%
- **LLM 生成**：DeepSeek 生成 300-400 字个性化推荐
- **引用来源追踪**：推荐理由带 `[1][2][3]` 编号，每条可追溯到图谱原文
- **多轮对话**：支持上下文追问（"那如果我加仓呢？"）

### 💡 4 条核心推荐规则

| 规则 | 触发条件 | 推荐逻辑 |
|------|---------|---------|
| 市场事件驱动 | 央行加息、股市波动 | 事件 → 影响产品 → 风险匹配客户 |
| 存款到期推荐 | 定期存款即将到期 | 到期资金 → 推荐替代产品 |
| 行为驱动推荐 | 浏览产品但未购买 | 浏览行为 → 推荐相似产品 |
| 闲钱理财推荐 | 闲钱 > 10 万 | 闲钱规模 → 风险匹配产品 |

### 🧠 GNN 异构图神经网络

- **模型架构**：特征投影 + 消息传递 + 嵌入输出
- **训练目标**：BPR（Bayesian Personalized Ranking）损失
- **评估结果**：测试集 AUC > 0.7
- **嵌入维度**：64 维客户/产品向量

---

## 🛠️ 技术栈

| 类别 | 技术 | 用途 |
|------|------|------|
| 数据处理 | pandas, numpy | 数据清洗、特征工程 |
| 图谱构建 | NetworkX | 异构图建模 |
| 深度学习 | PyTorch | GNN 模型训练 |
| 向量检索 | FAISS | 高效相似度检索 |
| 嵌入模型 | BGE-M3 | 1024 维文本向量 |
| 重排模型 | BGE-reranker-v2-m3 | 二次排序 |
| LLM 接口 | OpenAI SDK (DeepSeek) | 实体抽取、可解释推荐 |
| Web 应用 | Streamlit, Plotly | 交互式仪表盘 |
| 图谱可视化 | Pyvis | 交互式网络图 |

---

## 🚀 快速开始

### 基础功能（规则引擎 + GNN）

```bash
# 1. 克隆仓库
git clone https://gitee.com/ycai814/bank-recommendation-system.git
cd bank-recommendation-system

# 2. 安装依赖
pip install -r requirements.txt

# 3. 生成模拟数据（5000 客户 + 2 万节点）
python generate_bank_data.py

# 4. 构建知识图谱
python build_full_kg.py

# 5. 运行规则引擎推荐
python rule_recommend.py

# 6. 训练 GNN 模型
python export_to_gnn.py    # 导出数据
python gnn_model_pyg.py    # 训练模型
python gnn_recommend.py    # 生成推荐

# 7. 启动 Streamlit 应用
python -m streamlit run app_bank.py   # 规则引擎版
python -m streamlit run app_gnn.py    # GNN 推荐版

RAG-FLARKO 智能顾问（进阶）
# 1. 安装 RAG 相关依赖
pip install faiss-cpu sentence-transformers FlagEmbedding

# 2. 下载 BGE-M3 嵌入模型（约 2.2 GB）
# 国内推荐用 hf-mirror 镜像
$env:HF_ENDPOINT = "https://hf-mirror.com"
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"

# 3. 下载 Reranker 模型（约 2.2 GB，推荐用 ModelScope）
pip install modelscope
python download_reranker.py

# 4. 构建向量索引（约 7 分钟）
python build_vector_index.py

# 5. 测试多阶段检索器
python rag_retriever.py

# 6. 测试 LLM 生成模块
python rag_generator.py

# 7. 启动智能顾问应用
python -m streamlit run app_rag.py