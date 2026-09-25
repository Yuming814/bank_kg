import sys
sys.stdout.reconfigure(encoding='utf-8')

import pickle
import json

print("正在加载图谱...")
with open('bank_kg_full.pkl', 'rb') as f:
    G = pickle.load(f)
print(f"✅ 图谱加载完成\n")


# ================== 规则1：市场事件驱动 ==================
def rule_market_event(event_id, top_k=5):
    """事件 → MarketScope → 偏好客户 → 推荐产品"""
    if event_id not in G:
        return []
    
    event = G.nodes[event_id]
    print(f"\n{'='*70}")
    print(f"📢 规则1：市场事件驱动推荐")
    print(f"{'='*70}")
    print(f"事件: {event.get('title')}")
    print(f"情感: {event.get('sentiment_score')}")
    
    market_scopes = [v for u, v, d in G.edges(event_id, data=True)
                     if d.get('relation') == 'ABOUT']
    print(f"影响的 MarketScope: {market_scopes}")
    
    candidate_customers = set()
    for ms in market_scopes:
        for u, v, d in G.edges(data=True):
            if v == ms and d.get('relation') == 'PREFERS':
                candidate_customers.add(u)
    
    candidate_products = set()
    for ms in market_scopes:
        for u, v, d in G.edges(data=True):
            if v == ms and d.get('relation') == 'HAS_EXPOSURE_TO':
                candidate_products.add(u)
    
    print(f"候选客户: {len(candidate_customers)} 个")
    print(f"候选产品: {[G.nodes[p].get('name') for p in candidate_products]}")
    
    recommendations = []
    for cid in candidate_customers:
        cdata = G.nodes[cid]
        if cdata.get('last_login_days', 999) > 30:
            continue
        for pid in candidate_products:
            if G.has_edge(cid, pid):
                continue
            recommendations.append({
                'customer_id': cid,
                'customer_name': cdata.get('name'),
                'product_id': pid,
                'product_name': G.nodes[pid].get('name'),
                'event': event.get('title'),
                'reason': f"「{event.get('title')}」影响了你偏好的市场范围"
            })
    
    return recommendations[:top_k]


# ================== 规则2：存款到期推荐 ==================
def rule_deposit_maturity(top_k=5):
    """客户 → 资金事件(到期) → 偏好市场 → 推荐产品"""
    print(f"\n{'='*70}")
    print(f"💰 规则2：存款到期推荐")
    print(f"{'='*70}")
    
    recommendations = []
    
    for node, data in G.nodes(data=True):
        if data.get('node_type') != 'FundEvent':
            continue
        if data.get('event_type') != 'deposit_maturity':
            continue
        
        for u, v, d in G.edges(data=True):
            if v == node and d.get('relation') == 'TRIGGERS':
                customer_id = u
                customer = G.nodes[customer_id]
                amount = data.get('amount', 0)
                
                if amount < 50000:
                    continue
                
                preferred_scopes = [v for u2, v2, d2 in G.edges(customer_id, data=True)
                                    if d2.get('relation') == 'PREFERS']
                
                for ms in preferred_scopes:
                    for u2, v2, d2 in G.edges(data=True):
                        if v2 == ms and d2.get('relation') == 'HAS_EXPOSURE_TO':
                            pid = u2
                            if G.has_edge(customer_id, pid):
                                continue
                            recommendations.append({
                                'customer_id': customer_id,
                                'customer_name': customer.get('name'),
                                'amount': amount,
                                'maturity_date': data.get('expected_date'),
                                'product_id': pid,
                                'product_name': G.nodes[pid].get('name'),
                                'reason': f"存款{amount}元即将到期，推荐配置偏好产品"
                            })
    
    return recommendations[:top_k]


# ================== 规则3：行为驱动推荐 ==================
def rule_behavior_driven(top_k=5):
    """客户 → 行为(浏览) → 产品 → 同市场范围相似产品"""
    print(f"\n{'='*70}")
    print(f"👀 规则3：行为驱动推荐")
    print(f"{'='*70}")
    
    recommendations = []
    
    for node, data in G.nodes(data=True):
        if data.get('node_type') != 'BehaviorEvent':
            continue
        if data.get('behavior_type') != 'browse_product':
            continue
        
        target_products = [v for u, v, d in G.edges(node, data=True)
                          if d.get('relation') == 'TARGETS']
        
        for pid in target_products:
            market_scopes = [v for u, v, d in G.edges(pid, data=True)
                            if d.get('relation') == 'HAS_EXPOSURE_TO']
            
            for ms in market_scopes:
                similar_products = [u for u, v, d in G.edges(data=True)
                                   if v == ms and d.get('relation') == 'HAS_EXPOSURE_TO'
                                   and u != pid]
                
                for u, v, d in G.edges(data=True):
                    if v == node and d.get('relation') == 'PERFORMS':
                        customer_id = u
                        customer = G.nodes[customer_id]
                        
                        for similar_pid in similar_products:
                            if G.has_edge(customer_id, similar_pid):
                                continue
                            recommendations.append({
                                'customer_id': customer_id,
                                'customer_name': customer.get('name'),
                                'browsed_product': G.nodes[pid].get('name'),
                                'product_id': similar_pid,
                                'product_name': G.nodes[similar_pid].get('name'),
                                'reason': f"你浏览了「{G.nodes[pid].get('name')}」，推荐相似产品"
                            })
    
    return recommendations[:top_k]


# ================== 规则4：闲钱理财 ==================
def rule_idle_cash(top_k=5):
    """客户(闲钱>10万) → 偏好市场 → 推荐产品"""
    print(f"\n{'='*70}")
    print(f"💵 规则4：闲钱理财推荐")
    print(f"{'='*70}")
    
    recommendations = []
    
    for cid, cdata in G.nodes(data=True):
        if cdata.get('node_type') != 'Customer':
            continue
        
        idle_cash = cdata.get('idle_cash', 0)
        if idle_cash < 100000:
            continue
        
        preferred_scopes = [v for u, v, d in G.edges(cid, data=True)
                           if d.get('relation') == 'PREFERS']
        
        for ms in preferred_scopes:
            for u, v, d in G.edges(data=True):
                if v == ms and d.get('relation') == 'HAS_EXPOSURE_TO':
                    pid = u
                    if G.has_edge(cid, pid):
                        continue
                    recommendations.append({
                        'customer_id': cid,
                        'customer_name': cdata.get('name'),
                        'idle_cash': idle_cash,
                        'product_id': pid,
                        'product_name': G.nodes[pid].get('name'),
                        'reason': f"你有{idle_cash}元闲钱，推荐配置偏好产品"
                    })
    
    return recommendations[:top_k]


# ================== 执行所有规则 ==================
if __name__ == "__main__":
    print("=" * 70)
    print("🏦 银行产品推荐引擎 - 4条规则执行")
    print("=" * 70)
    
    recs1 = rule_market_event("EV_001", top_k=5)
    print(f"\n推荐结果 ({len(recs1)} 条):")
    for r in recs1:
        print(f"  → {r['customer_name']} ({r['customer_id']}): {r['product_name']}")
    
    recs2 = rule_deposit_maturity(top_k=5)
    print(f"\n推荐结果 ({len(recs2)} 条):")
    for r in recs2:
        print(f"  → {r['customer_name']}: 到期{r['amount']}元 → {r['product_name']}")
    
    recs3 = rule_behavior_driven(top_k=5)
    print(f"\n推荐结果 ({len(recs3)} 条):")
    for r in recs3:
        print(f"  → {r['customer_name']}: 浏览「{r['browsed_product']}」→ {r['product_name']}")
    
    recs4 = rule_idle_cash(top_k=5)
    print(f"\n推荐结果 ({len(recs4)} 条):")
    for r in recs4:
        print(f"  → {r['customer_name']}: 闲钱{r['idle_cash']}元 → {r['product_name']}")
    
    all_recs = {
        "market_event": recs1,
        "deposit_maturity": recs2,
        "behavior_driven": recs3,
        "idle_cash": recs4
    }
    with open("recommendations.json", "w", encoding="utf-8") as f:
        json.dump(all_recs, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*70}")
    print(f"✅ 所有规则执行完成，结果已保存到 recommendations.json")
    print(f"{'='*70}")