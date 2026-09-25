import sys
sys.stdout.reconfigure(encoding='utf-8')

import streamlit as st
import pickle
import json
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="GNN智能推荐", page_icon="🧠", layout="wide")
st.title("🧠 零售银行 - GNN 智能推荐系统")

@st.cache_data
def load_data():
    with open('gnn_data.pkl', 'rb') as f:
        data = pickle.load(f)
    with open('gnn_recommendations.json', 'r', encoding='utf-8') as f:
        recs = json.load(f)
    c_emb = np.load('customer_embeddings.npy')
    p_emb = np.load('product_embeddings.npy')
    return data, recs, c_emb, p_emb

data, recs, c_emb, p_emb = load_data()

# ============ 指标卡 ============
col1, col2, col3, col4 = st.columns(4)
col1.metric("👥 客户数", len(data['customer_ids']))
col2.metric("📦 产品数", len(data['product_ids']))
col3.metric("🔗 训练边", len(data['customer_product_edges']))
col4.metric("🧠 嵌入维度", c_emb.shape[1] if len(c_emb.shape) > 1 else 0)

st.markdown("---")

# ============ GNN 推荐结果 ============
st.subheader("🧠 GNN 推荐结果")

df_recs = []
for r in recs:
    for rec in r['recommendations']:
        df_recs.append({
            '客户ID': r['customer_id'],
            '推荐产品': rec['product_id'],
            '匹配得分': rec['score'],
        })

df = pd.DataFrame(df_recs)
if len(df) > 0:
    # Top 推荐产品
    top_products = df.groupby('推荐产品')['匹配得分'].mean().sort_values(ascending=False).reset_index()
    top_products.columns = ['产品', '平均得分']
    fig = px.bar(top_products, x='产品', y='平均得分',
                 color='平均得分', color_continuous_scale='Blues',
                 title='Top 推荐产品（平均得分）')
    st.plotly_chart(fig, use_container_width=True)
    
    # 明细
    st.dataframe(df, use_container_width=True)

# ============ 单客户查询 ============
st.markdown("---")
st.subheader("🔍 单客户推荐查询")

customer_list = data['customer_ids']
selected = st.selectbox("选择客户", customer_list, index=0)

if selected:
    c_idx = customer_list.index(selected)
    c_vec = c_emb[c_idx]
    
    # 计算得分
    scores = []
    for i, pid in enumerate(data['product_ids']):
        s = float(np.dot(c_vec, p_emb[i]))
        scores.append({'产品': pid, '匹配得分': s})
    
    df_scores = pd.DataFrame(scores).sort_values('匹配得分', ascending=False)
    
    fig = px.bar(df_scores, x='产品', y='匹配得分',
                 color='匹配得分', color_continuous_scale='Greens',
                 title=f'客户 {selected} 的产品匹配得分')
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df_scores, use_container_width=True)