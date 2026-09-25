import sys
sys.stdout.reconfigure(encoding='utf-8')  # 防止中文打印崩溃

import random
import json
from datetime import datetime, timedelta

random.seed(42)  # 固定随机种子，保证每次生成的数据一致

# ================== 1. 生成客户数据 ==================
print("正在生成客户数据...")
occupations = ["互联网从业者", "教师", "医生", "公务员", "工程师", "销售", "财务", "自由职业"]
risk_levels = ["保守型", "稳健型", "平衡型", "进取型", "激进型"]
income_levels = ["10-30w", "30-50w", "50-100w", "100w+"]

customers = []
for i in range(500):
    cid = f"C{10001+i}"
    customers.append({
        "customer_id": cid,
        "name": f"客户{cid}",
        "age": random.randint(25, 65),
        "occupation": random.choice(occupations),
        "income_level": random.choice(income_levels),
        "risk_tolerance": random.choice(risk_levels),
        "total_assets": random.randint(10, 500) * 10000,
        "deposit_balance": random.randint(5, 300) * 10000,
        "investment_balance": random.randint(0, 200) * 10000,
        "idle_cash": random.randint(0, 100) * 10000,
        "last_login_days": random.randint(0, 30),
        "deposit_maturity_date": (datetime.now() + timedelta(days=random.randint(-30, 180))).strftime('%Y-%m-%d')
    })
print(f"✅ 生成 {len(customers)} 个客户")

# ================== 2. 生成产品数据 ==================
print("正在生成产品数据...")
products = [
    {"product_id": "P001", "name": "活期存款", "type": "存款", "risk_level": "保守型", "yield_rate": 0.3, "min_amount": 0, "term_days": 0},
    {"product_id": "P002", "name": "1年期定期存款", "type": "存款", "risk_level": "保守型", "yield_rate": 1.8, "min_amount": 10000, "term_days": 365},
    {"product_id": "P003", "name": "3年期定期存款", "type": "存款", "risk_level": "保守型", "yield_rate": 2.75, "min_amount": 50000, "term_days": 1095},
    {"product_id": "P004", "name": "余额宝货币基金", "type": "货币基金", "risk_level": "保守型", "yield_rate": 2.1, "min_amount": 1, "term_days": 0},
    {"product_id": "P005", "name": "美元货币基金", "type": "货币基金", "risk_level": "稳健型", "yield_rate": 4.5, "min_amount": 1000, "term_days": 0},
    {"product_id": "P006", "name": "稳健债券基金", "type": "债券基金", "risk_level": "稳健型", "yield_rate": 3.8, "min_amount": 1000, "term_days": 0},
    {"product_id": "P007", "name": "中短债基金", "type": "债券基金", "risk_level": "稳健型", "yield_rate": 3.5, "min_amount": 100, "term_days": 0},
    {"product_id": "P008", "name": "平衡混合基金", "type": "混合基金", "risk_level": "平衡型", "yield_rate": 6.5, "min_amount": 1000, "term_days": 0},
    {"product_id": "P009", "name": "偏股混合基金", "type": "混合基金", "risk_level": "进取型", "yield_rate": 8.5, "min_amount": 1000, "term_days": 0},
    {"product_id": "P010", "name": "沪深300指数基金", "type": "股票基金", "risk_level": "进取型", "yield_rate": 10.0, "min_amount": 100, "term_days": 0},
    {"product_id": "P011", "name": "科技主题基金", "type": "股票基金", "risk_level": "激进型", "yield_rate": 15.0, "min_amount": 1000, "term_days": 0},
    {"product_id": "P012", "name": "重疾险", "type": "保险", "risk_level": "保守型", "yield_rate": 0, "min_amount": 5000, "term_days": 365},
    {"product_id": "P013", "name": "年金险", "type": "保险", "risk_level": "稳健型", "yield_rate": 3.0, "min_amount": 10000, "term_days": 3650},
]
print(f"✅ 生成 {len(products)} 个产品")

# ================== 3. 生成客户购买关系 ==================
print("正在生成购买关系...")
purchases = []
for customer in customers:
    n_products = random.randint(1, 4)
    suitable_products = [p for p in products if p['risk_level'] == customer['risk_tolerance']]
    if not suitable_products:
        suitable_products = [p for p in products if p['risk_level'] in ['保守型', '稳健型']]
    
    selected = random.sample(suitable_products, min(n_products, len(suitable_products)))
    for prod in selected:
        purchases.append({
            "customer_id": customer['customer_id'],
            "product_id": prod['product_id'],
            "amount": random.randint(1000, 100000),
            "purchase_date": (datetime.now() - timedelta(days=random.randint(1, 365))).strftime('%Y-%m-%d')
        })
print(f"✅ 生成 {len(purchases)} 条购买关系")

# ================== 4. 生成市场事件 ==================
print("正在生成市场事件...")
events = [
    {"event_id": "E001", "title": "央行宣布加息25个基点", "event_type": "政策监管", "sentiment_score": 60,
     "affected_products": ["P005", "P002"], "affected_industries": ["银行", "保险"]},
    {"event_id": "E002", "title": "A股大幅回调，沪深300跌3%", "event_type": "政策监管", "sentiment_score": -50,
     "affected_products": ["P006", "P007"], "affected_industries": ["证券"]},
    {"event_id": "E003", "title": "科技板块大涨，AI概念股集体涨停", "event_type": "产品发布", "sentiment_score": 80,
     "affected_products": ["P011", "P010"], "affected_industries": ["电子", "计算机"]},
    {"event_id": "E004", "title": "美联储降息预期升温", "event_type": "政策监管", "sentiment_score": 70,
     "affected_products": ["P005", "P008"], "affected_industries": ["银行"]},
    {"event_id": "E005", "title": "某大型房企债务违约", "event_type": "诉讼纠纷", "sentiment_score": -70,
     "affected_products": ["P006", "P007"], "affected_industries": ["房地产"]},
]
print(f"✅ 生成 {len(events)} 个市场事件")

# ================== 5. 生成资金事件 ==================
print("正在生成资金事件...")
fund_events = []
for customer in customers:
    if random.random() < 0.3:
        fund_events.append({
            "event_id": f"FUND_{customer['customer_id']}",
            "customer_id": customer['customer_id'],
            "event_type": "deposit_maturity",
            "amount": customer['deposit_balance'],
            "expected_date": customer['deposit_maturity_date'],
            "status": "upcoming"
        })
print(f"✅ 生成 {len(fund_events)} 个资金事件")

# ================== 6. 生成行为事件 ==================
print("正在生成行为事件...")
behavior_events = []
for customer in customers:
    for _ in range(random.randint(0, 5)):
        behavior_events.append({
            "event_id": f"BEH_{customer['customer_id']}_{random.randint(1, 9999)}",
            "customer_id": customer['customer_id'],
            "behavior_type": random.choice(["browse_product", "consult_service", "search"]),
            "target_product": random.choice(products)['product_id'],
            "timestamp": (datetime.now() - timedelta(days=random.randint(0, 30))).strftime('%Y-%m-%d %H:%M:%S'),
            "channel": random.choice(["APP", "WEB", "客服"])
        })
print(f"✅ 生成 {len(behavior_events)} 条行为事件")

# ================== 7. 保存所有数据 ==================
data = {
    "customers": customers,
    "products": products,
    "purchases": purchases,
    "events": events,
    "fund_events": fund_events,
    "behavior_events": behavior_events
}

with open("bank_data.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"\n{'='*60}")
print(f"✅ 模拟数据生成完成")
print(f"{'='*60}")
print(f"  客户: {len(customers)} 个")
print(f"  产品: {len(products)} 个")
print(f"  购买关系: {len(purchases)} 条")
print(f"  市场事件: {len(events)} 个")
print(f"  资金事件: {len(fund_events)} 个")
print(f"  行为事件: {len(behavior_events)} 条")
print(f"\n✅ 数据已保存到 bank_data.json")