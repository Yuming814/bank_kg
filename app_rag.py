import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import streamlit as st
import pickle
from rag_retriever import MultiStageRetriever
from rag_generator import generate_with_citations

# ================== 页面配置 ==================
st.set_page_config(
    page_title="智能投资顾问",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 零售银行 - RAG-FLARKO 智能顾问")
st.markdown("基于**知识图谱 + 大语言模型**的对话式投资顾问 | **推荐理由可追溯**")


# ================== 缓存检索器 ==================
@st.cache_resource
def get_retriever():
    return MultiStageRetriever()

retriever = get_retriever()


# ================== 加载客户列表 ==================
@st.cache_data
def load_data():
    with open('bank_kg_full.pkl', 'rb') as f:
        G = pickle.load(f)
    customers = [(n, d.get('name', n)) for n, d in G.nodes(data=True)
                 if d.get('node_type') == 'Customer']
    return G, customers

G, customers = load_data()
customer_names = [name for _, name in customers]
customer_id_map = {name: cid for cid, name in customers}


# ================== 侧边栏 ==================
st.sidebar.header("🔍 选择客户")
selected_name = st.sidebar.selectbox("客户列表", customer_names, index=0)
selected_cid = customer_id_map[selected_name]
cdata = G.nodes[selected_cid]

st.sidebar.markdown("---")
st.sidebar.markdown("### 👤 客户画像")
st.sidebar.write(f"**姓名**: {cdata.get('name')}")
st.sidebar.write(f"**年龄**: {cdata.get('age')} 岁")
st.sidebar.write(f"**职业**: {cdata.get('occupation')}")
st.sidebar.write(f"**风险偏好**: {cdata.get('risk_tolerance')}")
st.sidebar.write(f"**闲钱**: {cdata.get('idle_cash', 0):,} 元")

st.sidebar.markdown("---")
if st.sidebar.button("🔄 清除对话历史"):
    st.session_state.messages = []
    st.session_state.customer_id = selected_cid
    st.rerun()


# ================== 会话状态 ==================
if st.session_state.get('customer_id') != selected_cid:
    st.session_state.messages = []
    st.session_state.customer_id = selected_cid

if 'messages' not in st.session_state:
    st.session_state.messages = []


# ================== 主对话区 ==================
st.markdown("---")
st.subheader("💬 与您的智能投资顾问对话")

# 常见问题按钮
if len(st.session_state.messages) == 0:
    st.markdown("**常见问题**（点击快速提问）：")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📈 最近有什么投资机会？", use_container_width=True):
            st.session_state.quick_query = "最近有什么适合我的投资机会？"
    with col2:
        if st.button("🛡️ 如何优化资产配置？", use_container_width=True):
            st.session_state.quick_query = "我的资产配置是否需要优化？"
    with col3:
        if st.button("💵 闲钱怎么理财？", use_container_width=True):
            st.session_state.quick_query = "我有闲钱，推荐什么产品？"

# 显示历史消息
for msg in st.session_state.messages:
    if msg['role'] == 'user':
        with st.chat_message("user"):
            st.write(msg['content'])
    else:
        with st.chat_message("assistant", avatar="🧠"):
            st.markdown(msg['content'])
            # 展示引用来源
            if msg.get('citations'):
                with st.expander(f"📚 查看引用来源（共 {len(msg['citations'])} 条）"):
                    for i, c in enumerate(msg['citations']):
                        st.markdown(f"**[{i+1}]** {c['text']}")
                        st.caption(f"来源: {c['source']}")

# 处理快速提问
default_input = st.session_state.pop('quick_query', '')

# 输入框
user_query = st.chat_input("输入您的问题...")

if user_query:
    # 用户消息
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)
    
    # 生成回复
    with st.chat_message("assistant", avatar="🧠"):
        with st.spinner("🤔 AI 顾问正在分析..."):
            response, citations = generate_with_citations(
                selected_cid,
                user_query,
                retriever,
                chat_history=st.session_state.messages[:-1]
            )
        st.markdown(response)
        
        # 展示引用
        if citations:
            with st.expander(f"📚 查看引用来源（共 {len(citations)} 条）"):
                for i, c in enumerate(citations):
                    st.markdown(f"**[{i+1}]** {c['text']}")
                    st.caption(f"来源: {c['source']}")
    
    # 保存到历史
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "citations": citations
    })


# ================== 底部 ==================
st.markdown("---")
st.caption("📊 技术架构: RAG-FLARKO（多阶段知识图谱检索 + Reranker + LLM 生成） | 推荐理由可追溯")