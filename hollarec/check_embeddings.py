"""检查所有 embedding 文件的完整性"""
import torch
import sys
from pathlib import Path

def check_embedding_file(file_path, modality_name):
    """检查单个 embedding 文件"""
    print(f"\n检查 {modality_name}: {file_path}")
    try:
        # 加载文件
        data = torch.load(file_path, map_location='cpu', weights_only=True)
        
        if not isinstance(data, dict):
            print(f"  ❌ 错误: 不是字典类型,而是 {type(data)}")
            return False
        
        print(f"  ✓ 包含 {len(data)} 个 movies")
        
        # 检查每个 embedding
        corrupted = []
        for movie_id, emb in data.items():
            if not isinstance(emb, torch.Tensor):
                corrupted.append((movie_id, f"不是Tensor,而是{type(emb)}"))
                continue
            
            if torch.isnan(emb).any():
                corrupted.append((movie_id, "包含NaN"))
                continue
            
            if torch.isinf(emb).any():
                corrupted.append((movie_id, "包含Inf"))
                continue
        
        if corrupted:
            print(f"  ❌ 发现 {len(corrupted)} 个损坏的 embeddings:")
            for movie_id, reason in corrupted[:10]:  # 只显示前10个
                print(f"    - Movie {movie_id}: {reason}")
            if len(corrupted) > 10:
                print(f"    ... 还有 {len(corrupted) - 10} 个")
            return False
        else:
            print(f"  ✓ 所有 embeddings 正常")
            return True
            
    except Exception as e:
        print(f"  ❌ 加载失败: {type(e).__name__}: {e}")
        return False

def main():
    # base_path = Path("hollarec/data/dataset/redial/llava")
    base_path = Path("hollarec\\data\\dataset\\redial\\llava\\embeddings")
    
    files = {
        'txt': base_path / 'txt_embeddings.pt',
        'img': base_path / 'img_embeddings.pt', 
        'vdo': base_path / 'vdo_embeddings.pt',
        'ado': base_path / 'ado_embeddings.pt'
    }
    
    print("="*60)
    print("ReDial Embedding 完整性检查")
    print("="*60)
    
    all_ok = True
    for modality, file_path in files.items():
        if not file_path.exists():
            print(f"\n❌ {modality}: 文件不存在 {file_path}")
            all_ok = False
        else:
            if not check_embedding_file(file_path, modality):
                all_ok = False
    
    print("\n" + "="*60)
    if all_ok:
        print("✓ 所有 embedding 文件检查通过!")
    else:
        print("❌ 发现问题,请检查上述错误")
    print("="*60)
    
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())
