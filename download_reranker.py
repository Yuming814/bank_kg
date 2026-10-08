import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
from modelscope import snapshot_download

# 下载到项目目录下的 models 文件夹
PROJECT_MODEL_DIR = "./models"

print("=" * 70)
print("从 ModelScope 下载 BGE-reranker-v2-m3")
print("=" * 70)
print("模型大小约 2.27 GB，国内 CDN，预计 2-5 分钟\n")

try:
    model_dir = snapshot_download(
        'BAAI/bge-reranker-v2-m3',
        cache_dir=PROJECT_MODEL_DIR
    )
    
    print(f"\n✅ 模型下载完成")
    print(f"   路径: {model_dir}")
    
    # 检查关键文件是否存在
    print(f"\n📁 检查模型文件:")
    expected_files = [
        'config.json',
        'model.safetensors',
        'tokenizer.json',
        'tokenizer_config.json',
        'special_tokens_map.json',
        'sentencepiece.bpe.model'
    ]
    
    for f in expected_files:
        file_path = os.path.join(model_dir, f)
        if os.path.exists(file_path):
            size_mb = os.path.getsize(file_path) / 1024 / 1024
            print(f"  ✅ {f} ({size_mb:.1f} MB)")
        else:
            print(f"  ❌ {f} (缺失!)")
    
    print(f"\n📌 下一步:")
    print(f"   1. 确认 rag_retriever.py 的 use_reranker=True")
    print(f"   2. 运行: python rag_retriever.py")
    
except Exception as e:
    print(f"\n❌ 下载失败: {type(e).__name__} - {e}")
    print(f"\n💡 备用方案: 使用 hf-mirror 镜像重新下载")
    print(f"   $env:HF_ENDPOINT = 'https://hf-mirror.com'")
    print(f"   python -c \"from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-v2-m3')\"")