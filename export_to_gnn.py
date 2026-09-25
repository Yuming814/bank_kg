import sys
sys.stdout.reconfigure(encoding='utf-8')

import pickle
import json
import numpy as np
import pandas as pd

print("正在加载图谱...")
with open('bank_kg_full.pkl', 'rb') as f:
    G = pickle.load(f)

print(f"✅ 图谱加载完成: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")

# ================== 1. 提取节点列表和索引映射 ==================
print("\n[1/5] 提取节点索引映射...")

# 客户节点
customers = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'Customer']
customer_to_idx = {cid: i for i, cid in enumerate(customers)}

# 产品节点
products = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'FundProduct']
product_to_idx = {pid: i for i, pid in enumerate(products)}

# 事件节点（市场事件）
events = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'Event']
event_to_idx = {eid: i for i, eid in enumerate(events)}

# 市场范围节点
market_scopes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'MarketScope']
ms_to_idx = {ms: i for i, ms in enumerate(market_scopes)}

print(f"  客户: {len(customers)}")
print(f"  产品: {len(products)}")
print(f"  事件: {len(events)}")
print(f"  市场范围: {len(market_scopes)}")

# ================== 2. 提取边 ==================
print("\n[2/5] 提取边...")

# 客户-产品（HOLDS）
customer_product_edges = []
for u, v, d in G.edges(data=True):
    if d.get('relation') == 'HOLDS' and u in customer_to_idx and v in product_to_idx:
        customer_product_edges.append([
            customer_to_idx[u],
            product_to_idx[v],
            d.get('amount', 0),
        ])
print(f"  客户-产品 (HOLDS): {len(customer_product_edges)} 条")

# 事件-产品（AFFECTS，从Event的ABOUT链路上推断）
event_product_edges = []
for ev_id, ev_data in G.nodes(data=True):
    if ev_data.get('node_type') != 'Event':
        continue
    # 事件→MarketScope→产品
    for u, ms, d in G.edges(ev_id, data=True):
        if d.get('relation') != 'ABOUT':
            continue
        for pid, _, d2 in G.edges(data=True):
            if _ == ms and d2.get('relation') == 'HAS_EXPOSURE_TO':
                if ev_id in event_to_idx and pid in product_to_idx:
                    event_product_edges.append([
                        event_to_idx[ev_id],
                        product_to_idx[pid],
                    ])
print(f"  事件-产品 (AFFECTS): {len(event_product_edges)} 条")

# 客户-事件（PREFERS，通过市场范围关联）
customer_event_edges = []
for cid in customers:
    for ev_id in events:
        # 客户偏好的市场范围 ∩ 事件影响的市场范围
        c_scopes = set(v for u, v, d in G.edges(cid, data=True) if d.get('relation') == 'PREFERS')
        e_scopes = set(v for u, v, d in G.edges(ev_id, data=True) if d.get('relation') == 'ABOUT')
        if c_scopes & e_scopes:
            customer_event_edges.append([
                customer_to_idx[cid],
                event_to_idx[ev_id],
            ])
print(f"  客户-事件 (PREFERS): {len(customer_event_edges)} 条")

# ================== 3. 特征工程 ==================
print("\n[3/5] 构建节点特征...")

# ---------- 3.1 客户特征 ----------
customer_features = []
for cid in customers:
    cdata = G.nodes[cid]
    
    # 数值特征
    age = cdata.get('age', 35) / 100.0  # 归一化
    idle_cash = cdata.get('idle_cash', 0) / 1000000.0  # 归一化
    last_login = cdata.get('last_login_days', 30) / 30.0  # 归一化
    
    # 风险偏好 One-Hot
    risk = cdata.get('risk_tolerance', '稳健型')
    risk_onehot = [
        1 if risk == '保守型' else 0,
        1 if risk == '稳健型' else 0,
        1 if risk == '平衡型' else 0,
        1 if risk == '进取型' else 0,
        1 if risk == '激进型' else 0,
    ]
    
    # 收入 One-Hot
    income = cdata.get('income_level', '30-50w')
    income_onehot = [
        1 if income == '10-30w' else 0,
        1 if income == '30-50w' else 0,
        1 if income == '50-100w' else 0,
        1 if income == '100w+' else 0,
    ]
    
    customer_features.append([age, idle_cash, last_login] + risk_onehot + income_onehot)

customer_features = np.array(customer_features, dtype=np.float32)
print(f"  客户特征维度: {customer_features.shape}")

# ---------- 3.2 产品特征 ----------
product_features = []
for pid in products:
    pdata = G.nodes[pid]
    
    # 类型 One-Hot
    ptype = pdata.get('name', '')
    type_onehot = [
        1 if '存款' in ptype or '货币' in ptype else 0,
        1 if '债券' in ptype else 0,
        1 if '混合' in ptype else 0,
        1 if '股票' in ptype or '科技' in ptype else 0,
        1 if '黄金' in ptype else 0,
    ]
    
    product_features.append(type_onehot)

product_features = np.array(product_features, dtype=np.float32)
print(f"  产品特征维度: {product_features.shape}")

# ---------- 3.3 事件特征 ----------
event_features = []
for eid in events:
    edata = G.nodes[eid]
    sentiment = edata.get('sentiment_score', 0) / 100.0  # 归一化
    event_features.append([sentiment])

event_features = np.array(event_features, dtype=np.float32)
print(f"  事件特征维度: {event_features.shape}")

# ================== 4. 保存数据 ==================
print("\n[4/5] 保存数据...")

output = {
    'customer_ids': customers,
    'product_ids': products,
    'event_ids': events,
    'customer_features': customer_features,
    'product_features': product_features,
    'event_features': event_features,
    'customer_product_edges': customer_product_edges,
    'event_product_edges': event_product_edges,
    'customer_event_edges': customer_event_edges,
}

with open('gnn_data.pkl', 'wb') as f:
    pickle.dump(output, f)

print(f"  ✅ 数据已保存到 gnn_data.pkl")

# ================== 5. 汇总 ==================
print(f"\n{'='*60}")
print(f"✅ GNN 数据导出完成")
print(f"{'='*60}")
print(f"  节点数:")
print(f"    客户: {len(customers)}")
print(f"    产品: {len(products)}")
print(f"    事件: {len(events)}")
print(f"  边数:")
print(f"    客户-产品: {len(customer_product_edges)}")
print(f"    事件-产品: {len(event_product_edges)}")
print(f"    客户-事件: {len(customer_event_edges)}")