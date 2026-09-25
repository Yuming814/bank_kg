\# 🏦 零售银行 - 事件驱动产品推荐系统



> 基于知识图谱 + 规则引擎 + 图神经网络（GNN）的银行产品推荐系统



\[!\[Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)

\[!\[PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

\[!\[NetworkX](https://img.shields.io/badge/NetworkX-3.0+-green.svg)](https://networkx.org/)

\[!\[Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io/)



\---



\## 📖 项目简介



零售银行的产品推荐面临三大挑战：



1\. \*\*客户维度缺失\*\*：传统推荐只看"产品-产品"关联，忽略客户个性化需求

2\. \*\*事件驱动不足\*\*：市场事件（加息、股市波动）无法自动触发推荐

3\. \*\*可解释性差\*\*：黑盒模型无法解释"为什么推荐这个产品"



本项目构建了一个 \*\*"事件-产品-客户"三位一体\*\*的知识图谱，并结合\*\*规则引擎\*\*和\*\*图神经网络（GNN）\*\*两套推荐方案，实现了从市场事件到客户推荐的完整闭环。



\---



\## 🏗️ 系统架构



```

┌─────────────────────────────────────────────────────┐

│                    数据层 (Data Layer)                │

│  客户360°画像 + 产品库 + 购买记录 + 行为日志 + 市场事件  │

└─────────────────────────────────────────────────────┘

&#x20;                         ↓

┌─────────────────────────────────────────────────────┐

│              知识图谱层 (Knowledge Graph Layer)        │

│  6 层异构图谱: 产品维度 / 市场范围 / 持仓 / 事件 / 客户 / 行为 │

│  1925 节点, 3000+ 边                                  │

└─────────────────────────────────────────────────────┘

&#x20;                         ↓

┌─────────────────────────────────────────────────────┐

│             推荐引擎层 (Recommendation Layer)          │

│  ① 规则引擎（4条规则）  ② GNN 异构图神经网络           │

└─────────────────────────────────────────────────────┘

&#x20;                         ↓

┌─────────────────────────────────────────────────────┐

│              应用层 (Application Layer)               │

│  Streamlit 交互式仪表盘 + 智能顾问对话                 │

└─────────────────────────────────────────────────────┘

```



\---



\## ✨ 核心特性



\### 📊 知识图谱建模



\- \*\*6 层异构图谱\*\*：产品维度 → 市场范围 → 产品持仓 → 事件 → 客户 → 行为

\- \*\*14 类节点\*\*：Customer、FundProduct、Event、MarketScope、AssetClass 等

\- \*\*15 种关系\*\*：HOLDS、PREFERS、ABOUT、HAS\_EXPOSURE\_TO、TRIGGERS 等



\### 🎯 双推荐引擎



| 引擎 | 特点 | 适用场景 |

|------|------|---------|

| \*\*规则引擎\*\* | 4 条业务规则，可解释，响应快 | 快速上线，规则明确 |

| \*\*GNN 模型\*\* | 自动学习高阶关系，个性化 | 复杂场景，智能推荐 |



\### 💡 4 条核心推荐规则



| 规则 | 触发条件 | 推荐逻辑 |

|------|---------|---------|

| 市场事件驱动 | 央行加息、股市波动 | 事件 → 影响产品 → 风险匹配客户 |

| 存款到期推荐 | 定期存款即将到期 | 到期资金 → 推荐替代产品 |

| 行为驱动推荐 | 浏览产品但未购买 | 浏览行为 → 推荐相似产品 |

| 闲钱理财推荐 | 闲钱 > 10 万 | 闲钱规模 → 风险匹配产品 |



\### 🧠 GNN 异构图神经网络



\- \*\*模型架构\*\*：特征投影 + 消息传递 + 嵌入输出

\- \*\*训练目标\*\*：BPR（Bayesian Personalized Ranking）损失

\- \*\*评估结果\*\*：测试集 AUC > 0.7

\- \*\*嵌入维度\*\*：64 维客户/产品向量



\---



\## 🛠️ 技术栈



| 类别 | 技术 | 用途 |

|------|------|------|

| 数据处理 | pandas, numpy | 数据清洗、特征工程 |

| 图谱构建 | NetworkX | 异构图建模 |

| 深度学习 | PyTorch | GNN 模型训练 |

| Web 应用 | Streamlit, Plotly | 交互式仪表盘 |

| 图谱可视化 | Pyvis | 交互式网络图 |

| LLM 接口 | OpenAI SDK (DeepSeek) | 实体抽取、可解释推荐 |



\---



\## 🚀 快速开始



\### 1. 克隆仓库



```bash

git clone https://gitee.com/ycai814/bank-recommendation-system.git

cd bank-recommendation-system

```



\### 2. 安装依赖



```bash

pip install -r requirements.txt

```



\### 3. 生成模拟数据



```bash

python generate\_bank\_data.py

```



\### 4. 构建知识图谱



```bash

python build\_full\_kg.py

```



\### 5. 运行规则引擎推荐



```bash

python rule\_recommend.py

```



\### 6. 训练 GNN 模型



```bash

python export\_to\_gnn.py    # 导出数据

python gnn\_model\_pyg.py    # 训练模型

python gnn\_recommend.py    # 生成推荐

```



\### 7. 启动 Streamlit 应用



```bash

\# 规则引擎版

python -m streamlit run app\_bank.py



\# GNN 推荐版

python -m streamlit run app\_gnn.py

```



浏览器打开 `http://localhost:8501` 即可访问。



\---



\## 📁 项目结构



```

bank-recommendation-system/

├── README.md                    # 项目说明

├── requirements.txt             # 依赖清单

├── .gitignore                   # Git 忽略配置

├── docs/

│   └── PROJECT\_REPORT.md        # 完整项目报告

├── generate\_bank\_data.py        # 生成模拟数据

├── build\_full\_kg.py             # 构建知识图谱

├── verify\_kg.py                 # 验证图谱结构

├── rule\_recommend.py            # 规则引擎推荐

├── app\_bank.py                  # Streamlit（规则引擎版）

├── export\_to\_gnn.py             # 导出 GNN 数据

├── gnn\_model\_pyg.py             # GNN 模型训练

├── gnn\_recommend.py             # GNN 推荐生成

├── test\_pyg.py                  # 环境自检

└── app\_gnn.py                   # Streamlit（GNN 版）

```



\---



\## 📊 核心数据



\### 数据规模



| 指标 | 数值 |

|------|------|

| 客户数 | 500 |

| 产品数 | 13 |

| 购买关系 | 903 |

| 行为事件 | 1200+ |

| 资金事件 | 150+ |

| 图谱节点 | 1925 |

| 图谱边 | 3000+ |



\### 客户画像维度



\- \*\*静态属性\*\*：年龄、职业、收入水平

\- \*\*风险偏好\*\*：保守型 / 稳健型 / 平衡型 / 进取型 / 激进型

\- \*\*资产状况\*\*：总资产、存款、投资、闲钱

\- \*\*行为特征\*\*：登录频率、浏览产品、咨询客服



\---



\## 🖼️ 项目截图



\### 规则引擎推荐界面



!\[规则引擎](docs/images/rule\_engine.png)



\### GNN 智能推荐界面



!\[GNN 推荐](docs/images/gnn\_recommend.png)



> 📌 截图说明：将你的应用截图保存到 `docs/images/` 目录，命名对应即可。



\---



\## 🎯 技术难点



| 难点 | 解决方案 |

|------|---------|

| \*\*客户-产品多对多关系\*\* | 用异构图建模，而非扁平表格 |

| \*\*事件影响链路复杂\*\* | 通过 MarketScope 中间层解耦 |

| \*\*规则引擎僵硬\*\* | 引入 GNN 自动学习高阶关系 |

| \*\*冷启动问题\*\* | 用 PREFERS 关系通过风险偏好匹配 |

| \*\*数据隐私\*\* | 全部使用模拟数据，不涉及真实客户信息 |



\---



\## 📈 未来方向



\### 短期

\- \[ ] 扩充产品库到 100+ 产品

\- \[ ] 增加客户行为序列特征

\- \[ ] 完善 README 截图和演示



\### 中期

\- \[ ] 升级 GNN 为 HGT Transformer

\- \[ ] 集成 RAG-FLARKO 实现 LLM 可解释推荐

\- \[ ] A/B 测试对比规则 vs GNN 效果



\### 长期

\- \[ ] 接入真实银行数据（脱敏）

\- \[ ] 实时推荐服务（Kafka + Flink）

\- \[ ] 多目标优化（收益 + 风险 + 客户满意度）



\---



\## 📚 参考资源



\- \[PyTorch Geometric 官方文档](https://pytorch-geometric.readthedocs.io/)

\- \[NetworkX 官方文档](https://networkx.org/documentation/stable/)

\- \[Streamlit 官方文档](https://docs.streamlit.io/)

\- \[RAG-FLARKO 论文](https://arxiv.org/abs/2511.11583)



\---



\## 📄 许可证



本项目仅用于学习和研究目的。



\---



\## 👤 作者



\- \*\*Gitee\*\*: \[@ycai814](https://gitee.com/ycai814)

\- \*\*项目地址\*\*: \[bank-recommendation-system](https://gitee.com/ycai814/bank-recommendation-system)



\---



⭐ 如果这个项目对你有帮助，欢迎 Star！

