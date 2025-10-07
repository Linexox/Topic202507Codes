import os
import cv2
import torch
import clip
from PIL import Image
import numpy as np
from tqdm import tqdm

# 初始化CLIP模型
device = "cuda:6" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.enabled = False
model, preprocess = clip.load("ViT-B/32", device=device)

def extract_four_keyframes(video_path):
    """
    从视频中均匀提取4个关键帧
    :param video_path: 视频文件路径
    :return: PIL图像列表
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return []
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = [int(frame_count * (i*2+1)/8) for i in range(4)]  # 均匀分布的四帧
    
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
    
    cap.release()
    return frames

def encode_and_concatenate(frames):
    """
    编码图像并拼接嵌入向量
    :param frames: PIL图像列表
    :return: 拼接后的嵌入向量(torch.Tensor)
    """
    embeddings = []
    for frame in frames:
        image = preprocess(frame).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image)
        embeddings.append(image_features.cpu())
    
    # 拼接所有嵌入 (4 x 512 -> 2048)
    return torch.cat(embeddings, dim=1).squeeze(0)  # 形状: [2048]

def process_trailers_folder(trailers_folder, output_dir):
    """
    处理整个文件夹中的预告片
    :param trailers_folder: 包含预告片的文件夹路径
    :param output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有视频文件
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    video_files = [f for f in os.listdir(trailers_folder) 
                  if f.lower().endswith(video_extensions)]
    
    for video_file in tqdm(video_files, desc="Processing trailers"):
        video_path = os.path.join(trailers_folder, video_file)
        
        # 提取4个关键帧
        frames = extract_four_keyframes(video_path)
        if not frames:
            continue
            
        # 编码并拼接嵌入
        concatenated_embedding = encode_and_concatenate(frames)
        
        # 保存拼接后的嵌入
        output_name = os.path.splitext(video_file)[0] + ".pt"
        output_path = os.path.join(output_dir, output_name)
        torch.save(concatenated_embedding, output_path)

if __name__ == "__main__":
    trailers_folder = "./video"
    output_dir = "./vdo_emb"
    process_trailers_folder(trailers_folder, output_dir)