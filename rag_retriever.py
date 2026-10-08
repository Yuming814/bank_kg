import sys
sys.stdout.reconfigure(encoding='utf-8')

# 强制 CPU 模式（避免 GPU 报错）
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from reranker import Reranker


class MultiStageRetriever:
    """RAG-FLARKO 多阶段检索器（含 Reranker 二次排序）"""
    
    def __init__(self, kg_path='bank_kg_full.pkl',
                 index_path='kg_vector_index.faiss',
                 meta_path='kg_vector_metadata.pkl',
                 use_reranker=True):
        
        # 1. 加载图谱
        print("正在加载知识图谱...")
        with open(kg_path, 'rb') as f:
            self.G = pickle.load(f)
        
        # 2. 加载 FAISS 索引和元数据
        print("正在加载向量索引...")
        self.index = faiss.read_index(index_path)
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        self.node_ids = meta['node_ids']
        self.node_texts = meta['node_texts']
        self.node_types = meta['node_types']
        
        # 3. 加载嵌入模型（强制 CPU）
        print("正在加载 BGE-M3 嵌入模型...")
        import glob

        cache_dir = os.path.expanduser("~/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots")
        if os.path.exists(cache_dir):
            snapshots = glob.glob(os.path.join(cache_dir, "*"))
            if snapshots:
                model_path = snapshots[0]  # 取第一个（通常只有一个）
            else:
                model_path = 'BAAI/bge-m3'
        else:
            model_path = 'BAAI/bge-m3'

        print(f"使用模型路径: {model_path}")
        self.model = SentenceTransformer(model_path, device='cpu')

        
        # 4. 加载 Reranker（可选）
        self.use_reranker = use_reranker
        if use_reranker:
            print("正在加载 Reranker...")
            self.reranker = Reranker()
        else:
            self.reranker = None
        
        print(f"✅ 检索器初始化完成: {len(self.node_ids)} 个节点 (Reranker: {'开启' if use_reranker else '关闭'})")

    def semantic_search(self, query, top_k=10, filter_type=None, use_rerank=True):
        """
        语义搜索（两阶段：向量召回 + Reranker 重排）
        
        Args:
            query: 查询文本
            top_k: 返回 Top-K
            filter_type: 只返回该类型的节点（如 'Event'、'FundProduct'）
            use_rerank: 是否使用 Reranker 重排
        """
        # ========== 阶段1：向量召回 ==========
        # 如果用 Reranker，召回更多候选（50 个）；否则只召回 top_k
        recall_k = 50 if (self.use_reranker and use_rerank) else top_k
        
        query_vec = self.model.encode(
            [query],
            normalize_embeddings=True,
            device='cpu'
        ).astype('float32')
        
        # 如果指定了 filter_type，多召回一些以便过滤
        search_k = recall_k * 3 if filter_type else recall_k
        distances, indices = self.index.search(query_vec, search_k)
        
        candidates = []
        for i, idx in enumerate(indices[0]):
            if idx == -1:
                continue
            node_id = self.node_ids[idx]
            node_type = self.node_types[idx]
            
            # 类型过滤
            if filter_type and node_type != filter_type:
                continue
            
            candidates.append({
                'node_id': node_id,
                'node_type': node_type,
                'text': self.node_texts[idx],
                'score': float(distances[0][i])
            })
            
            if len(candidates) >= recall_k:
                break
        
        # ========== 阶段2：Reranker 重排 ==========
        if self.use_reranker and use_rerank and len(candidates) > top_k:
            candidates = self.reranker.rerank(query, candidates, top_k=top_k)
        else:
            candidates = candidates[:top_k]
        
        return candidates

    def retrieve_personal_context(self, customer_id, query, top_k=5):
        """
        阶段一：从个人知识图谱 (PKG) 中检索客户行为相关实体。
        直接查图谱（不依赖语义搜索），保证精准。
        """
        if customer_id not in self.G:
            return []
        
        pkg_entities = []
        
        # 1. 客户持有产品
        for u, v, d in self.G.edges(customer_id, data=True):
            if d.get('relation') == 'HOLDS':
                product_data = self.G.nodes[v]
                pkg_entities.append({
                    'node_id': v,
                    'node_type': 'FundProduct',
                    'text': f"客户持有: {product_data.get('name', v)} (金额: {d.get('amount', 0)}元)",
                    'source': 'PKG_HOLDS'
                })
        
        # 2. 客户偏好市场范围
        for u, v, d in self.G.edges(customer_id, data=True):
            if d.get('relation') == 'PREFERS':
                pkg_entities.append({
                    'node_id': v,
                    'node_type': 'MarketScope',
                    'text': f"客户偏好市场范围: {v}",
                    'source': 'PKG_PREFERS'
                })
        
        # 3. 客户近期行为（最近3条）
        behaviors = []
        for u, v, d in self.G.edges(customer_id, data=True):
            if d.get('relation') == 'PERFORMS':
                behaviors.append(v)
        
        for beh in behaviors[-3:]:
            beh_data = self.G.nodes[beh]
            pkg_entities.append({
                'node_id': beh,
                'node_type': 'BehaviorEvent',
                'text': f"近期行为: {beh_data.get('behavior_type', '')} @ {beh_data.get('timestamp', '')}",
                'source': 'PKG_BEHAVIOR'
            })
        
        return pkg_entities[:top_k]

    def retrieve_market_context(self, pkg_entities, query, top_k=10):
        """
        阶段二：基于 PKG 上下文，从市场知识图谱 (MKG) 中检索相关信号。
        用语义搜索 + 图谱邻域扩展结合。
        """
        mkg_entities = []
        relevant_scopes = set()
        
        # 从 PKG 实体中找出关联的 MarketScope
        for entity in pkg_entities:
            entity_id = entity['node_id']
            
            if entity['node_type'] == 'MarketScope':
                relevant_scopes.add(entity_id)
                continue
            
            if entity_id in self.G:
                for u, v, d in self.G.edges(entity_id, data=True):
                    if d.get('relation') == 'HAS_EXPOSURE_TO':
                        relevant_scopes.add(v)
                for u, v, d in self.G.edges(data=True):
                    if u == entity_id and d.get('relation') == 'HAS_EXPOSURE_TO':
                        relevant_scopes.add(v)
        
        # 基于 MarketScope 检索市场事件和产品
        seen_ids = set()
        for scope in relevant_scopes:
            # 相关市场事件
            for u, v, d in self.G.edges(data=True):
                if v == scope and d.get('relation') == 'ABOUT':
                    if u in seen_ids:
                        continue
                    seen_ids.add(u)
                    event_data = self.G.nodes[u]
                    mkg_entities.append({
                        'node_id': u,
                        'node_type': 'Event',
                        'text': f"相关市场事件: {event_data.get('title', '')} (情感得分: {event_data.get('sentiment_score', 0)})",
                        'source': 'MKG_EVENT'
                    })
            
            # 同市场范围的其他产品
            for u, v, d in self.G.edges(data=True):
                if v == scope and d.get('relation') == 'HAS_EXPOSURE_TO':
                    if u in seen_ids:
                        continue
                    seen_ids.add(u)
                    product_data = self.G.nodes[u]
                    mkg_entities.append({
                        'node_id': u,
                        'node_type': 'FundProduct',
                        'text': f"相关产品: {product_data.get('name', '')}",
                        'source': 'MKG_PRODUCT'
                    })
        
        # 补充：用语义搜索 + Reranker 找相关事件
        try:
            semantic_events = self.semantic_search(
                query, 
                top_k=3, 
                filter_type='Event',
                use_rerank=True
            )
            for e in semantic_events:
                if e['node_id'] not in seen_ids:
                    seen_ids.add(e['node_id'])
                    mkg_entities.append({
                        'node_id': e['node_id'],
                        'node_type': 'Event',
                        'text': f"语义相关事件: {e['text']}",
                        'source': 'MKG_SEMANTIC'
                    })
        except Exception as e:
            print(f"  ⚠️ 语义搜索失败: {e}")
        
        return mkg_entities[:top_k]

    def retrieve_full_context(self, customer_id, query):
        """完整的两阶段检索"""
        pkg_context = self.retrieve_personal_context(customer_id, query)
        mkg_context = self.retrieve_market_context(pkg_context, query)
        return {
            'pkg': pkg_context,
            'mkg': mkg_context
        }


# ================== 测试 ==================
if __name__ == "__main__":
    print("=" * 70)
    print("RAG-FLARKO 检索器测试（含 Reranker）")
    print("=" * 70)
    
    retriever = MultiStageRetriever(use_reranker=True)
    
    customer_id = "C10001"
    query = "最近有什么适合我的投资机会？"
    
    print(f"\n{'='*70}")
    print(f"客户 {customer_id} 的查询: {query}")
    print(f"{'='*70}")
    
    print("\n【阶段一】从个人知识图谱 (PKG) 检索...")
    pkg_context = retriever.retrieve_personal_context(customer_id, query)
    for e in pkg_context:
        print(f"  → {e['text']} [{e['source']}]")
    
    print("\n【阶段二】基于 PKG 上下文，从市场知识图谱 (MKG) 检索...")
    mkg_context = retriever.retrieve_market_context(pkg_context, query)
    for e in mkg_context:
        print(f"  → {e['text']} [{e['source']}]")
    
    print("\n" + "=" * 70)
    print("【Reranker 效果对比测试】")
    print("=" * 70)
    
    test_query = "稳健型债券基金"
    
    print(f"\n查询: {test_query}")
    print("\n【只用向量检索（不用 Reranker）】")
    results_raw = retriever.semantic_search(test_query, top_k=5, use_rerank=False)
    for r in results_raw:
        print(f"  [{r['score']:.4f}] {r['text'][:80]}")
    
    print("\n【向量检索 + Reranker 重排】")
    results_reranked = retriever.semantic_search(test_query, top_k=5, use_rerank=True)
    for r in results_reranked:
        rerank_score = r.get('rerank_score', 'N/A')
        if isinstance(rerank_score, float):
            print(f"  [Rerank: {rerank_score:.4f}] {r['text'][:80]}")
        else:
            print(f"  {r['text'][:80]}")
    
    print("\n✅ 检索器测试完成")