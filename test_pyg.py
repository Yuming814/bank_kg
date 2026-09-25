import sys
sys.stdout.reconfigure(encoding='utf-8')

try:
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
except ImportError:
    print("❌ PyTorch 未安装")
    sys.exit(1)

try:
    import torch_geometric
    print(f"✅ PyG: {torch_geometric.__version__}")
    PYG_AVAILABLE = True
except ImportError:
    print("⚠️ PyG 未安装，将使用纯 PyTorch 版本")
    PYG_AVAILABLE = False

print(f"\n环境检查完成")
print(f"使用 {'PyG' if PYG_AVAILABLE else '纯 PyTorch'} 版本")