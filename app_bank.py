import sys
sys.stdout.reconfigure(encoding='utf-8')

import streamlit as st
import pickle
import json
import pandas as pd
import plotly.express as px
from collections import Counter

st.set_page_config(page_title="银行客户产品推荐", page_icon="🏦", layout="wide")
st.title("🏦 零售银行 - 事件驱动产品推荐系统")

@st.cache_data
def load_data():
    with open('bank_kg_full.pkl', 'rb') as f:
        G = pickle.load(f)
    with open('recommendations.json', 'r', encoding='utf-8') as f:
        recs = json.load(f)
    return G, recs

G, recs = load_data()

# ============ 顶部指标 ============
col1, col2, col3, col4 = st.columns(4)
customers = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'Customer']
products = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'FundProduct']
col1.metric("👥 客户数", len(customers))
col2.metric("📦 产品数", len(products))
col3.metric("🔗 图谱节点", G.number_of_nodes())
col4.metric("📊 推荐总数", sum(len(v) for v in recs.values()))

st.markdown("---")

# ============ 推荐结果展示 ============
st.subheader("📋 推荐结果")

tab1, tab2, tab3, tab4 = st.tabs(["📢 市场事件驱动", "💰 存款到期", "👀 行为驱动", "💵 闲钱理财"])

with tab1:
    df = pd.DataFrame(recs['market_event'])
    if len(df) > 0:
        st.dataframe(df[['customer_name', 'customer_id', 'product_name', 'event', 'reason']], use_container_width=True)
    else:
        st.info("暂无推荐")

with tab2:
    df = pd.DataFrame(recs['deposit_maturity'])
    if len(df) > 0:
        st.dataframe(df[['customer_name', 'amount', 'maturity_date', 'product_name', 'reason']], use_container_width=True)

with tab3:
    df = pd.DataFrame(recs['behavior_driven'])
    if len(df) > 0:
        st.dataframe(df[['customer_name', 'browsed_product', 'product_name', 'reason']], use_container_width=True)

with tab4:
    df = pd.DataFrame(recs['idle_cash'])
    if len(df) > 0:
        st.dataframe(df[['customer_name', 'idle_cash', 'product_name', 'reason']], use_container_width=True)

# ============ 图谱结构可视化 ============
st.markdown("---")
st.subheader("🕸️ 图谱节点类型分布")

node_types = Counter(d.get('node_type', 'Unknown') for _, d in G.nodes(data=True))
fig = px.bar(x=list(node_types.keys()), y=list(node_types.values()),
             labels={'x': '节点类型', 'y': '数量'},
             color=list(node_types.values()),
             color_continuous_scale='Blues')
st.plotly_chart(fig, use_container_width=True)

# ============ 客户查询 ============
st.markdown("---")
st.subheader("🔍 客户推荐查询")

customer_list = sorted([G.nodes[c].get('name', c) for c in customers])
selected = st.selectbox("选择客户", customer_list, index=0)

selected_id = None
for c in customers:
    if G.nodes[c].get('name') == selected:
        selected_id = c
        break

if selected_id:
    cdata = G.nodes[selected_id]
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("年龄", cdata.get('age'))
    col_b.metric("职业", cdata.get('occupation'))
    col_c.metric("风险偏好", cdata.get('risk_tolerance'))
    col_d.metric("闲钱", f"{cdata.get('idle_cash', 0):,}元")
    
    holds = [(v, G.nodes[v].get('name')) for u, v, d in G.edges(selected_id, data=True)
             if d.get('relation') == 'HOLDS']
    st.write(f"**持有产品**: {', '.join([h[1] for h in holds])}")
    
    prefers = [v for u, v, d in G.edges(selected_id, data=True)
               if d.get('relation') == 'PREFERS']
    st.write(f"**偏好市场范围**: {', '.join(prefers)}")