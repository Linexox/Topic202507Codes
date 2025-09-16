import torch
import clip
from PIL import Image
import os
from tqdm import tqdm  # 可选，用于进度条

# 设置设备
device = "cuda:7" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.enabled = False

# 加载CLIP模型
model, preprocess = clip.load("ViT-B/32", device=device)  # 使用ViT-B/32模型
model.eval()

# 设置路径
image_folder = "./image"  # 替换为你的海报文件夹路径
output_folder = "./img_emb"    # 替换为你想保存嵌入的文件夹路径

# 创建输出文件夹（如果不存在）
os.makedirs(output_folder, exist_ok=True)

# 获取所有jpg文件
image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]

# 处理每张图片并保存嵌入
for image_file in tqdm(image_files, desc="Processing images"):
    # 构建完整路径
    image_path = os.path.join(image_folder, image_file)
    
    try:
        # 加载和预处理图像
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        
        # 获取图像嵌入
        with torch.no_grad():
            image_features = model.encode_image(image)
            
            # 保存为pt文件
            output_path = os.path.join(output_folder, f"{os.path.splitext(image_file)[0]}.pt")
            torch.save(image_features.cpu(), output_path)
            
    except Exception as e:
        print(f"Error processing {image_file}: {str(e)}")

print("所有图像处理完成！")