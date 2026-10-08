import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from openai import OpenAI
from rag_retriever import MultiStageRetriever

# 初始化 DeepSeek 客户端
client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY", "sk-your-api-key-here"),
    base_url="https://api.deepseek.com",
    timeout=60.0
)


def generate_with_citations(customer_id, query, retriever, chat_history=None):
    """
    带引用来源的推荐生成
    返回: (推荐文本, 引用来源列表)
    """
    if chat_history is None:
        chat_history = []
    
    # 1. 多阶段检索
    context = retriever.retrieve_full_context(customer_id, query)
    pkg_context = context['pkg']
    mkg_context = context['mkg']
    
    if not pkg_context and not mkg_context:
        return f"抱歉，没有找到客户 {customer_id} 的相关信息。", []
    
    # 2. 客户画像
    customer_data = retriever.G.nodes.get(customer_id, {})
    customer_profile = (
        f"姓名: {customer_data.get('name', '')}, "
        f"年龄: {customer_data.get('age', '')}, "
        f"职业: {customer_data.get('occupation', '')}, "
        f"风险偏好: {customer_data.get('risk_tolerance', '')}, "
        f"闲钱: {customer_data.get('idle_cash', 0)}元"
    )
    
    # 3. 构建带编号的上下文（关键！）
    all_context = []
    for e in pkg_context:
        all_context.append(e)
    for e in mkg_context:
        all_context.append(e)
    
    context_with_ids = "\n".join([
        f"[{i+1}] {e['text']} (来源: {e['source']})"
        for i, e in enumerate(all_context)
    ])
    
    # 4. System Prompt（要求引用编号）
    system_prompt = f"""你是一位专业的银行客户经理，需要为客户生成个性化的投资推荐。

【客户画像】
{customer_profile}

【可参考的信息（共 {len(all_context)} 条）】
{context_with_ids}

【回答要求】
1. 结合客户个人情况和市场环境，推荐 1-2 个最合适的产品。
2. **每条推荐理由必须用 [编号] 标明引用了哪条信息**。
   例如："考虑到您的风险偏好[1]，建议配置..."
3. 如果有合适的配置比例建议，请一并给出。
4. 用专业、亲切的语气，控制在 300-400 字。"""
    
    # 5. 构建完整消息列表
    messages = [{"role": "system", "content": system_prompt}]
    
    # 添加历史对话（多轮）
    for msg in chat_history[-4:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    
    messages.append({"role": "user", "content": query})
    
    # 6. 调用 LLM
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0.7,
            max_tokens=700
        )
        return response.choices[0].message.content, all_context
    except Exception as e:
        return f"生成推荐时出错: {type(e).__name__} - {e}", []


if __name__ == "__main__":
    retriever = MultiStageRetriever()
    
    customer_id = "C10001"
    query = "最近有什么适合我的投资机会？"
    
    print(f"\n{'='*70}")
    print(f"客户 {customer_id} 的查询: {query}")
    print(f"{'='*70}\n")
    
    recommendation, citations = generate_with_citations(customer_id, query, retriever)
    
    print("【推荐】")
    print(recommendation)
    
    print(f"\n{'='*70}")
    print(f"【引用来源（共 {len(citations)} 条）】")
    print(f"{'='*70}")
    for i, c in enumerate(citations):
        print(f"  [{i+1}] {c['text']}")
        print(f"      来源: {c['source']}\n")