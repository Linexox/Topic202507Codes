import pandas as pd
import torch
from transformers import T5Tokenizer, T5Model
import os
from tqdm import tqdm
import csv

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(file_path: str) -> pd.DataFrame:
    """加载CSV数据"""
    try:
        df = pd.read_csv(file_path)
        print(f"成功加载数据，共{len(df)}条记录")
        return df
    except FileNotFoundError:
        print(f"错误：文件 {file_path} 不存在")
        return pd.DataFrame()
    except Exception as e:
        print(f"错误：加载文件时出错 - {e}")
        return pd.DataFrame()

def initialize_model() -> tuple:
    """初始化模型和分词器"""
    print("正在加载模型和分词器...")
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    model = T5Model.from_pretrained('t5-base')
    model = model.to(device)
    model.eval()
    print("模型加载完成")
    return tokenizer, model

def extract_features(text: str, tokenizer, model) -> torch.Tensor:
    """从文本中提取特征"""
    # 编码文本
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 只使用编码器获取特征
    with torch.no_grad():
        # 使用encoder而不是完整模型
        encoder_outputs = model.encoder(**inputs)
        # 使用编码器最后一层的隐藏状态
        features = encoder_outputs.last_hidden_state[:, 0, :]
    
    # 如果需要1024维特征但模型输出不足，可以进行填充或其他处理
    if features.shape[1] < 1024:
        features = torch.nn.functional.pad(features, (0, 1024 - features.shape[1]))
    elif features.shape[1] > 1024:
        features = features[:, :1024]
    
    return features.cpu()

def save_features(movie_id: str, features: torch.Tensor, output_dir: str = "movie_features"):
    """保存特征到.pt文件"""
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{movie_id}.pt")
    
    # 保存特征
    torch.save(features, file_path)
    # print(f"特征已保存到 {file_path}")

def main():
    # 文件路径
    output_dir = "txt_emb"
    
    # 加载数据
    mv={}
    csv_path = os.path.join('./', 'movies_with_mentions.csv')
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) < 2:
                continue
            movie_id, movie_name = row[0], row[1]
            mv[movie_id] = movie_name
    tokenizer, model = initialize_model()
    
    # 处理每部电影
    print("开始提取特征...")
    for movie_id in tqdm(mv):
        movie_name = mv[movie_id]
        print(movie_id, movie_name)
        features = extract_features(movie_name, tokenizer, model)
        save_features(movie_id, features, output_dir)
    
    print(f"特征提取完成，所有文件已保存到 {output_dir} 目录")

if __name__ == "__main__":
    main()