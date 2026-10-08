import sys
sys.stdout.reconfigure(encoding='utf-8')

# ================== 离线 + CPU 模式 ==================
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import glob
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class Reranker:
    """BGE-reranker 二次排序器（纯 Transformers 实现，绕过 FlagEmbedding）"""
    
    def __init__(self, model_name=None):
        if model_name is None:
            model_name = self._find_local_cache()
        
        print(f"正在加载 Reranker: {model_name}...")
        
        try:
            # 加载 tokenizer 和 model
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                local_files_only=True
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                local_files_only=True,
                torch_dtype=torch.float32  # CPU 用 fp32
            )
            self.model.eval()  # 评估模式
            self.model.to('cpu')
            
            print("✅ Reranker 加载完成")
            self.available = True
        except Exception as e:
            print(f"⚠️ Reranker 加载失败: {type(e).__name__} - {e}")
            print("💡 提示: Reranker 将被禁用，检索器会退化为纯向量检索")
            self.tokenizer = None
            self.model = None
            self.available = False
    
    def _find_local_cache(self):
        """按优先级查找 Reranker 模型"""
        # 优先级1：项目目录
        project_paths = [
            "./models/BAAI/bge-reranker-v2-m3",
            "./models/bge-reranker-v2-m3",
        ]
        for p in project_paths:
            if os.path.exists(p) and os.path.exists(os.path.join(p, 'model.safetensors')):
                print(f"  ✅ 使用项目目录模型: {p}")
                return p
        
        # 优先级2：HF 缓存
        candidates = ["BAAI/bge-reranker-v2-m3", "BAAI/bge-reranker-base"]
        hf_cache = os.path.expanduser("~/.cache/huggingface/hub")
        
        for model_id in candidates:
            cache_dir_name = "models--" + model_id.replace("/", "--")
            snapshots_dir = os.path.join(hf_cache, cache_dir_name, "snapshots")
            
            if os.path.exists(snapshots_dir):
                snapshots = glob.glob(os.path.join(snapshots_dir, "*"))
                for snap in snapshots:
                    if os.path.exists(os.path.join(snap, 'model.safetensors')):
                        print(f"  ✅ 找到 HF 缓存: {snap}")
                        return snap
        
        print("  ⚠️ 未找到本地缓存，将尝试下载")
        return "BAAI/bge-reranker-v2-m3"
    
    def rerank(self, query, candidates, top_k=10):
        """对候选文档进行重排（纯 Transformers 实现）"""
        if not candidates:
            return []
        
        if not self.available or self.model is None:
            return candidates[:top_k]
        
        try:
            pairs = [(query, c['text']) for c in candidates]
            
            # Tokenize
            inputs = self.tokenizer(
                [list(p) for p in pairs],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            
            # 推理
            with torch.no_grad():
                outputs = self.model(**inputs)
                scores = outputs.logits.view(-1).float().cpu().numpy()
            
            # 赋值
            for i, c in enumerate(candidates):
                c['rerank_score'] = float(scores[i])
            
            # 排序
            candidates_sorted = sorted(
                candidates,
                key=lambda x: x['rerank_score'],
                reverse=True
            )
            
            return candidates_sorted[:top_k]
        
        except Exception as e:
            print(f"  ⚠️ Reranker 计算失败: {type(e).__name__} - {e}")
            return candidates[:top_k]


# ================== 测试 ==================
if __name__ == "__main__":
    print("=" * 70)
    print("Reranker 测试（纯 Transformers 实现）")
    print("=" * 70)
    
    reranker = Reranker()
    
    if not reranker.available:
        print("\n⚠️ Reranker 不可用，测试跳过")
        sys.exit(0)
    
    query = "适合稳健型客户的产品"
    candidates = [
        {'node_id': 'P001', 'text': '产品: 工银稳健债券基金, 风险: 稳健型', 'score': 0.65},
        {'node_id': 'P002', 'text': '产品: 科技主题基金, 风险: 激进型', 'score': 0.50},
        {'node_id': 'P003', 'text': '产品: 招银货币基金, 风险: 保守型', 'score': 0.45},
        {'node_id': 'P004', 'text': '客户: 客户C10001, 教师, 65岁', 'score': 0.30},
    ]
    
    print("\n【重排前】")
    for c in candidates:
        print(f"  [{c['score']:.4f}] {c['text']}")
    
    results = reranker.rerank(query, candidates, top_k=4)
    
    print(f"\n【重排后】查询: {query}")
    for r in results:
        score = r.get('rerank_score', 'N/A')
        if isinstance(score, float):
            print(f"  [Rerank: {score:.4f}] {r['text']}")
        else:
            print(f"  {r['text']}")
    
    print("\n✅ Reranker 测试完成")