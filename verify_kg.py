import sys
sys.stdout.reconfigure(encoding='utf-8')

import pickle
import networkx as nx

print("正在加载图谱...")
with open('bank_kg_full.pkl', 'rb') as f:
    G = pickle.load(f)

print(f"✅ 图谱加载完成: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
print("=" * 70)

# ================== 1. 客户邻域验证 ==================
cid = "C10001"
print(f"\n📌 客户 {cid} 的完整图谱邻域:")
print(f"{'='*70}")

if cid not in G:
    print(f"❌ 客户 {cid} 不存在")
else:
    cdata = G.nodes[cid]
    print(f"\n客户属性:")
    for key in ['name', 'age', 'occupation', 'risk_tolerance', 'idle_cash', 'last_login_days']:
        print(f"  {key}: {cdata.get(key)}")

    # 持有产品
    holds = [(v, G.nodes[v].get('name')) for u, v, d in G.edges(cid, data=True)
             if d.get('relation') == 'HOLDS']
    print(f"\n持有产品 ({len(holds)} 个):")
    for pid, name in holds:
        print(f"  → {name} ({pid})")

    # 偏好市场范围
    prefers = [v for u, v, d in G.edges(cid, data=True)
               if d.get('relation') == 'PREFERS']
    print(f"\n偏好市场范围 ({len(prefers)} 个):")
    for ms in prefers:
        print(f"  → {ms}")

    # 资金事件
    fund_events = [v for u, v, d in G.edges(cid, data=True)
                   if d.get('relation') == 'TRIGGERS']
    print(f"\n资金事件 ({len(fund_events)} 个):")
    for fe in fund_events[:3]:
        fedata = G.nodes[fe]
        print(f"  → {fe}: {fedata.get('event_type')} {fedata.get('amount')}元, 到期{fedata.get('expected_date')}")

    # 行为事件
    behaviors = [v for u, v, d in G.edges(cid, data=True)
                 if d.get('relation') == 'PERFORMS']
    print(f"\n行为事件 ({len(behaviors)} 个，显示前5条):")
    for be in behaviors[:5]:
        bedata = G.nodes[be]
        print(f"  → {bedata.get('behavior_type')} @ {bedata.get('timestamp')}")


# ================== 2. 产品反向验证 ==================
print(f"\n{'='*70}")
print("📌 产品反向验证：哪些客户持有该产品")
print(f"{'='*70}")

for fp_id in ['FP_001', 'FP_002', 'FP_003', 'FP_004', 'FP_005']:
    if fp_id not in G:
        continue
    holders = [u for u, v, d in G.edges(data=True)
               if v == fp_id and d.get('relation') == 'HOLDS']
    print(f"  {fp_id} ({G.nodes[fp_id].get('name')}): {len(holders)} 个客户持有")


# ================== 3. 事件链路验证 ==================
print(f"\n{'='*70}")
print("📌 事件链路验证：从事件到客户")
print(f"{'='*70}")

for ev_id in ['EV_001', 'EV_002', 'EV_003', 'EV_004']:
    if ev_id not in G:
        continue
    evdata = G.nodes[ev_id]
    print(f"\n事件 {ev_id}: {evdata.get('title')}")
    print(f"  情感: {evdata.get('sentiment_score')}")

    # ABOUT → MarketScope
    about = [v for u, v, d in G.edges(ev_id, data=True)
             if d.get('relation') == 'ABOUT']
    print(f"  ABOUT → MarketScope: {about}")

    for ms in about:
        # 暴露产品
        products = [u for u, v, d in G.edges(data=True)
                    if v == ms and d.get('relation') == 'HAS_EXPOSURE_TO']
        # 偏好客户
        customers = [u for u, v, d in G.edges(data=True)
                     if v == ms and d.get('relation') == 'PREFERS']
        print(f"    MarketScope {ms}:")
        print(f"      暴露产品: {[G.nodes[p].get('name') for p in products]}")
        print(f"      偏好客户数: {len(customers)}")


# ================== 4. 全局统计 ==================
print(f"\n{'='*70}")
print("📌 全局图谱统计")
print(f"{'='*70}")

from collections import Counter
node_types = Counter(d.get('node_type', 'Unknown') for _, d in G.nodes(data=True))
print("\n节点类型分布:")
for ntype, count in node_types.most_common():
    print(f"  {ntype}: {count}")

edge_types = Counter(d.get('relation', 'Unknown') for _, _, d in G.edges(data=True))
print("\n关系类型分布:")
for etype, count in edge_types.most_common():
    print(f"  {etype}: {count}")

print(f"\n✅ 图谱结构验证完成")