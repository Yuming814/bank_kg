import sys
sys.stdout.reconfigure(encoding='utf-8')

import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score

# ================== 1. 加载数据 ==================
print("正在加载 GNN 数据...")
with open('gnn_data.pkl', 'rb') as f:
    data = pickle.load(f)

customer_features = torch.tensor(data['customer_features'])
product_features = torch.tensor(data['product_features'])
event_features = torch.tensor(data['event_features'])

print(f"客户特征: {customer_features.shape}")
print(f"产品特征: {product_features.shape}")
print(f"事件特征: {event_features.shape}")


# ================== 2. 定义简化版异构图模型 ==================
class HeteroGNN(nn.Module):
    """
    简化版异构图神经网络：
    1. 分别对每种节点做特征投影
    2. 通过消息传递聚合邻域信息
    3. 输出客户和产品的嵌入
    """
    
    def __init__(self, customer_dim, product_dim, event_dim, hidden_dim=64):
        super().__init__()
        
        # 特征投影层
        self.customer_proj = nn.Linear(customer_dim, hidden_dim)
        self.product_proj = nn.Linear(product_dim, hidden_dim)
        self.event_proj = nn.Linear(event_dim, hidden_dim)
        
        # 消息传递层（简化版：直接聚合）
        self.cp_agg = nn.Linear(hidden_dim, hidden_dim)  # 客户-产品聚合
        self.ep_agg = nn.Linear(hidden_dim, hidden_dim)  # 事件-产品聚合
        self.ce_agg = nn.Linear(hidden_dim, hidden_dim)  # 客户-事件聚合
        
        # 输出层
        self.customer_out = nn.Linear(hidden_dim, hidden_dim)
        self.product_out = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, customer_feat, product_feat, event_feat,
                cp_edges, ep_edges, ce_edges):
        """
        cp_edges: [num_cp_edges, 2] (customer_idx, product_idx)
        ep_edges: [num_ep_edges, 2] (event_idx, product_idx)
        ce_edges: [num_ce_edges, 2] (customer_idx, event_idx)
        """
        # 1. 特征投影
        c_emb = self.customer_proj(customer_feat)
        p_emb = self.product_proj(product_feat)
        e_emb = self.event_proj(event_feat)
        
        # 2. 消息传递（简化版）
        # 2.1 客户聚合：从持有的产品获取信息
        if len(cp_edges) > 0:
            c_idx, p_idx = cp_edges[:, 0], cp_edges[:, 1]
            # 按客户聚合其持有的产品嵌入
            c_agg = torch.zeros_like(c_emb)
            c_agg.index_add_(0, c_idx, p_emb[p_idx])
            # 归一化（除以度）
            c_deg = torch.zeros(c_emb.size(0))
            c_deg.index_add_(0, c_idx, torch.ones(len(c_idx)))
            c_deg = c_deg.clamp(min=1).unsqueeze(1)
            c_agg = c_agg / c_deg
            c_emb = c_emb + self.cp_agg(c_agg)
        
        # 2.2 产品聚合：从事件和客户获取信息
        if len(ep_edges) > 0:
            e_idx, p_idx = ep_edges[:, 0], ep_edges[:, 1]
            p_agg = torch.zeros_like(p_emb)
            p_agg.index_add_(0, p_idx, e_emb[e_idx])
            p_deg = torch.zeros(p_emb.size(0))
            p_deg.index_add_(0, p_idx, torch.ones(len(p_idx)))
            p_deg = p_deg.clamp(min=1).unsqueeze(1)
            p_agg = p_agg / p_deg
            p_emb = p_emb + self.ep_agg(p_agg)
        
        # 2.3 客户聚合：从事件获取信息
        if len(ce_edges) > 0:
            c_idx, e_idx = ce_edges[:, 0], ce_edges[:, 1]
            c_agg = torch.zeros_like(c_emb)
            c_agg.index_add_(0, c_idx, e_emb[e_idx])
            c_deg = torch.zeros(c_emb.size(0))
            c_deg.index_add_(0, c_idx, torch.ones(len(c_idx)))
            c_deg = c_deg.clamp(min=1).unsqueeze(1)
            c_agg = c_agg / c_deg
            c_emb = c_emb + self.ce_agg(c_agg)
        
        # 3. 输出层
        c_out = self.customer_out(c_emb)
        p_out = self.product_out(p_emb)
        
        return c_out, p_out


# ================== 3. 准备训练数据 ==================
print("\n正在准备训练数据...")

# 边数据
cp_edges = torch.tensor([e[:2] for e in data['customer_product_edges']], dtype=torch.long)
ep_edges = torch.tensor(data['event_product_edges'], dtype=torch.long) if data['event_product_edges'] else torch.zeros((0, 2), dtype=torch.long)
ce_edges = torch.tensor(data['customer_event_edges'], dtype=torch.long) if data['customer_event_edges'] else torch.zeros((0, 2), dtype=torch.long)

print(f"客户-产品边: {cp_edges.shape}")
print(f"事件-产品边: {ep_edges.shape}")
print(f"客户-事件边: {ce_edges.shape}")

# 划分训练/测试集
n_edges = len(cp_edges)
indices = torch.randperm(n_edges)
train_size = int(n_edges * 0.8)

train_edges = cp_edges[indices[:train_size]]
test_edges = cp_edges[indices[train_size:]]

print(f"训练边: {len(train_edges)}, 测试边: {len(test_edges)}")


# ================== 4. 训练 ==================
print("\n开始训练...")

model = HeteroGNN(
    customer_dim=customer_features.size(1),
    product_dim=product_features.size(1),
    event_dim=event_features.size(1),
    hidden_dim=64
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

n_customers = customer_features.size(0)
n_products = product_features.size(0)

for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    
    # 前向传播
    c_out, p_out = model(customer_features, product_features, event_features,
                        train_edges, ep_edges, ce_edges)
    
    # 正样本得分（内积）
    pos_scores = (c_out[train_edges[:, 0]] * p_out[train_edges[:, 1]]).sum(dim=1)
    
    # 负样本（随机采样）
    n_neg = len(train_edges)
    neg_c = torch.randint(0, n_customers, (n_neg,))
    neg_p = torch.randint(0, n_products, (n_neg,))
    neg_scores = (c_out[neg_c] * p_out[neg_p]).sum(dim=1)
    
    # BPR损失
    loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()
    
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1}/100, Loss: {loss.item():.4f}")


# ================== 5. 评估 ==================
print("\n评估模型...")
model.eval()
with torch.no_grad():
    c_out, p_out = model(customer_features, product_features, event_features,
                         cp_edges, ep_edges, ce_edges)
    
    # 正样本
    pos_scores = (c_out[test_edges[:, 0]] * p_out[test_edges[:, 1]]).sum(dim=1).numpy()
    
    # 负样本
    n_neg = len(test_edges)
    neg_c = torch.randint(0, n_customers, (n_neg,))
    neg_p = torch.randint(0, n_products, (n_neg,))
    neg_scores = (c_out[neg_c] * p_out[neg_p]).sum(dim=1).numpy()
    
    # AUC
    y_true = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    y_score = np.concatenate([pos_scores, neg_scores])
    auc = roc_auc_score(y_true, y_score)
    print(f"\n✅ 测试集 AUC: {auc:.4f}")

# 保存模型和嵌入
torch.save(model.state_dict(), 'gnn_model.pt')
np.save('customer_embeddings.npy', c_out.numpy())
np.save('product_embeddings.npy', p_out.numpy())
print(f"\n✅ 模型已保存到 gnn_model.pt")
print(f"✅ 客户嵌入已保存到 customer_embeddings.npy")
print(f"✅ 产品嵌入已保存到 product_embeddings.npy")