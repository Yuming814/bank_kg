import sys
sys.stdout.reconfigure(encoding='utf-8')

import json
import pickle
import networkx as nx
from collections import Counter

print("正在加载数据...")
with open("bank_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

G = nx.DiGraph()

# ================== 第1层：产品维度 ==================
print("\n[第1层] 构建产品维度节点...")

asset_classes = [
    {"id": "AC_EQUITY", "name": "股票"},
    {"id": "AC_FIXED_INCOME", "name": "固定收益"},
    {"id": "AC_CASH", "name": "现金"},
    {"id": "AC_REAL_ESTATE", "name": "房地产"},
    {"id": "AC_COMMODITIES", "name": "商品"},
    {"id": "AC_EQUITY_CN", "name": "中国股票", "parent": "AC_EQUITY"},
    {"id": "AC_EQUITY_US", "name": "美国股票", "parent": "AC_EQUITY"},
    {"id": "AC_BOND_GOV", "name": "国债", "parent": "AC_FIXED_INCOME"},
    {"id": "AC_BOND_CORP", "name": "企业债", "parent": "AC_FIXED_INCOME"},
]
for ac in asset_classes:
    G.add_node(ac['id'], node_type='AssetClass', name=ac['name'])
    if ac.get('parent'):
        G.add_edge(ac['id'], ac['parent'], relation='IS_SUBCLASS_OF')

geo_scopes = [
    {"id": "GEO_CN", "name": "中国"},
    {"id": "GEO_US", "name": "美国"},
    {"id": "GEO_EU", "name": "欧洲"},
    {"id": "GEO_ASIA", "name": "亚洲（除中国）"},
    {"id": "GEO_GLOBAL", "name": "全球"},
]
for g in geo_scopes:
    G.add_node(g['id'], node_type='GeographicScope', name=g['name'])

industries = [
    {"id": "IND_TECH", "name": "信息技术"},
    {"id": "IND_FINANCE", "name": "金融"},
    {"id": "IND_HEALTH", "name": "医疗保健"},
    {"id": "IND_CONSUMER", "name": "消费"},
    {"id": "IND_ENERGY", "name": "能源"},
    {"id": "IND_MATERIALS", "name": "原材料"},
    {"id": "IND_INDUSTRIAL", "name": "工业"},
    {"id": "IND_UTILITY", "name": "公用事业"},
    {"id": "IND_REALESTATE", "name": "房地产"},
]
for ind in industries:
    G.add_node(ind['id'], node_type='IndustryCategory', name=ind['name'])

goods = [
    {"id": "GOODS_GOLD", "name": "黄金"},
    {"id": "GOODS_OIL", "name": "原油"},
    {"id": "GOODS_AGRI", "name": "农产品"},
]
for gd in goods:
    G.add_node(gd['id'], node_type='GoodsCategory', name=gd['name'])

print(f"  ✅ 资产类别: {len(asset_classes)} 个")
print(f"  ✅ 地理范围: {len(geo_scopes)} 个")
print(f"  ✅ 行业类别: {len(industries)} 个")
print(f"  ✅ 商品类别: {len(goods)} 个")


# ================== 第2层：MarketScope ==================
print("\n[第2层] 构建市场范围节点...")

market_scopes = [
    {"id": "MS_CN_EQUITY", "asset_class": "AC_EQUITY_CN", "geo": "GEO_CN", "industry": "IND_TECH"},
    {"id": "MS_US_EQUITY", "asset_class": "AC_EQUITY_US", "geo": "GEO_US", "industry": "IND_TECH"},
    {"id": "MS_CN_BOND", "asset_class": "AC_BOND_GOV", "geo": "GEO_CN"},
    {"id": "MS_CN_CASH", "asset_class": "AC_CASH", "geo": "GEO_CN"},
    {"id": "MS_GLOBAL_GOLD", "asset_class": "AC_COMMODITIES", "geo": "GEO_GLOBAL", "goods": "GOODS_GOLD"},
]
for ms in market_scopes:
    G.add_node(ms['id'], node_type='MarketScope', name=ms['id'])
    if ms.get('asset_class'):
        G.add_edge(ms['id'], ms['asset_class'], relation='ASSET_CLASS')
    if ms.get('geo'):
        G.add_edge(ms['id'], ms['geo'], relation='GEOGRAPHY')
    if ms.get('industry'):
        G.add_edge(ms['id'], ms['industry'], relation='INDUSTRY')
    if ms.get('goods'):
        G.add_edge(ms['id'], ms['goods'], relation='GOODS')

print(f"  ✅ 市场范围: {len(market_scopes)} 个")


# ================== 第3层：FundProduct + HoldingItem + Issuer ==================
print("\n[第3层] 构建产品与持仓节点...")

issuers = [
    {"id": "ISS_001", "name": "易方达基金"},
    {"id": "ISS_002", "name": "华夏基金"},
    {"id": "ISS_003", "name": "工商银行"},
    {"id": "ISS_004", "name": "招商银行"},
    {"id": "ISS_005", "name": "贵州茅台"},
    {"id": "ISS_006", "name": "宁德时代"},
]
for iss in issuers:
    G.add_node(iss['id'], node_type='Issuer', name=iss['name'])

holdings = [
    {"id": "HI_001", "name": "贵州茅台", "issuer": "ISS_005", "weight": 0.08},
    {"id": "HI_002", "name": "宁德时代", "issuer": "ISS_006", "weight": 0.06},
    {"id": "HI_003", "name": "招商银行", "issuer": "ISS_004", "weight": 0.05},
]
for h in holdings:
    G.add_node(h['id'], node_type='HoldingItem', name=h['name'], weight=h['weight'])
    G.add_edge(h['id'], h['issuer'], relation='HAS_ISSUER')

fund_products = [
    {"id": "FP_001", "name": "华夏科技成长基金", "market_scope": "MS_CN_EQUITY", "holdings": ["HI_001", "HI_002"]},
    {"id": "FP_002", "name": "易方达中概互联ETF", "market_scope": "MS_US_EQUITY", "holdings": ["HI_003"]},
    {"id": "FP_003", "name": "工银稳健债券基金", "market_scope": "MS_CN_BOND", "holdings": []},
    {"id": "FP_004", "name": "招银货币基金", "market_scope": "MS_CN_CASH", "holdings": []},
    {"id": "FP_005", "name": "黄金主题基金", "market_scope": "MS_GLOBAL_GOLD", "holdings": []},
]
for fp in fund_products:
    G.add_node(fp['id'], node_type='FundProduct', name=fp['name'])
    G.add_edge(fp['id'], fp['market_scope'], relation='HAS_EXPOSURE_TO')
    for hid in fp.get('holdings', []):
        G.add_edge(fp['id'], hid, relation='HAS_ITEM')

print(f"  ✅ 发行方: {len(issuers)} 个")
print(f"  ✅ 持仓项: {len(holdings)} 个")
print(f"  ✅ 基金产品: {len(fund_products)} 个")


# ================== 第4层：Event + EconomicIndicator ==================
print("\n[第4层] 构建事件与经济指标节点...")

indicators = [
    {"id": "EI_001", "name": "CPI同比", "value": 2.3},
    {"id": "EI_002", "name": "10年期国债收益率", "value": 2.85},
    {"id": "EI_003", "name": "美联储基准利率", "value": 5.25},
]
for ei in indicators:
    G.add_node(ei['id'], node_type='EconomicIndicator', name=ei['name'], value=ei['value'])

events = [
    {"id": "EV_001", "title": "央行加息25个基点", "event_type": "政策监管",
     "sentiment": 60, "market_scope": "MS_CN_CASH", "driver": "EI_002"},
    {"id": "EV_002", "title": "A股科技板块大涨3%", "event_type": "产品发布",
     "sentiment": 80, "market_scope": "MS_CN_EQUITY", "driver": "EI_001"},
    {"id": "EV_003", "title": "美联储降息预期升温", "event_type": "政策监管",
     "sentiment": 70, "market_scope": "MS_US_EQUITY", "driver": "EI_003"},
    {"id": "EV_004", "title": "黄金价格创历史新高", "event_type": "其他",
     "sentiment": 75, "market_scope": "MS_GLOBAL_GOLD", "driver": "EI_001"},
]
for ev in events:
    G.add_node(ev['id'], node_type='Event', title=ev['title'],
               event_type=ev['event_type'], sentiment_score=ev['sentiment'])
    G.add_edge(ev['id'], ev['market_scope'], relation='ABOUT')
    if ev.get('driver'):
        G.add_edge(ev['id'], ev['driver'], relation='HAS_DRIVER')

print(f"  ✅ 经济指标: {len(indicators)} 个")
print(f"  ✅ 事件: {len(events)} 个")


# ================== 第5层：Customer ==================
print("\n[第5层] 构建客户层节点...")

for c in data['customers']:
    G.add_node(
        c['customer_id'],
        node_type='Customer',
        name=c['name'],
        age=c['age'],
        occupation=c['occupation'],
        income_level=c['income_level'],
        risk_tolerance=c['risk_tolerance'],
        total_assets=c['total_assets'],
        deposit_balance=c['deposit_balance'],
        idle_cash=c['idle_cash'],
        last_login_days=c['last_login_days']
    )

risk_to_scopes = {
    "保守型": ["MS_CN_CASH", "MS_CN_BOND"],
    "稳健型": ["MS_CN_BOND", "MS_GLOBAL_GOLD"],
    "平衡型": ["MS_CN_EQUITY", "MS_CN_BOND"],
    "进取型": ["MS_CN_EQUITY", "MS_US_EQUITY"],
    "激进型": ["MS_CN_EQUITY", "MS_US_EQUITY", "MS_GLOBAL_GOLD"],
}

for c in data['customers']:
    cid = c['customer_id']
    preferred_scopes = risk_to_scopes.get(c['risk_tolerance'], [])
    for ms_id in preferred_scopes:
        G.add_edge(cid, ms_id, relation='PREFERS')

print(f"  ✅ {len(data['customers'])} 个客户节点")


# ================== 第6层：客户行为与持有 ==================
print("\n[第6层] 构建客户行为与持有关系...")

mapping = {"P001": "FP_004", "P002": "FP_003", "P003": "FP_003",
           "P004": "FP_004", "P005": "FP_004", "P006": "FP_003",
           "P007": "FP_003", "P008": "FP_001", "P009": "FP_001",
           "P010": "FP_002", "P011": "FP_001", "P012": "FP_004", "P013": "FP_003"}

holds_count = 0
for pur in data['purchases']:
    fp_id = mapping.get(pur['product_id'])
    if fp_id and fp_id in G:
        G.add_edge(pur['customer_id'], fp_id, relation='HOLDS',
                   amount=pur['amount'], date=pur['purchase_date'])
        holds_count += 1
print(f"  ✅ HOLDS 关系: {holds_count} 条")

fund_event_count = 0
for fe in data['fund_events']:
    G.add_node(fe['event_id'], node_type='FundEvent',
               event_type=fe['event_type'], amount=fe['amount'],
               expected_date=fe['expected_date'])
    G.add_edge(fe['customer_id'], fe['event_id'], relation='TRIGGERS')
    fund_event_count += 1
print(f"  ✅ FundEvent 节点: {fund_event_count} 个")

behavior_count = 0
for be in data['behavior_events']:
    G.add_node(be['event_id'], node_type='BehaviorEvent',
               behavior_type=be['behavior_type'],
               timestamp=be['timestamp'], channel=be['channel'])
    G.add_edge(be['customer_id'], be['event_id'], relation='PERFORMS')
    fp_id = mapping.get(be['target_product'])
    if fp_id and fp_id in G:
        G.add_edge(be['event_id'], fp_id, relation='TARGETS')
    behavior_count += 1
print(f"  ✅ BehaviorEvent 节点: {behavior_count} 个")


# ================== 保存 ==================
with open('bank_kg_full.pkl', 'wb') as f:
    pickle.dump(G, f)

print(f"\n{'='*60}")
print(f"✅ 完整银行知识图谱构建完成")
print(f"{'='*60}")
print(f"  总节点: {G.number_of_nodes()}")
print(f"  总边: {G.number_of_edges()}")

node_types = Counter(d.get('node_type', 'Unknown') for _, d in G.nodes(data=True))
print(f"\n节点类型分布:")
for ntype, count in node_types.most_common():
    print(f"  {ntype}: {count}")

edge_types = Counter(d.get('relation', 'Unknown') for _, _, d in G.edges(data=True))
print(f"\n关系类型分布:")
for etype, count in edge_types.most_common():
    print(f"  {etype}: {count}")