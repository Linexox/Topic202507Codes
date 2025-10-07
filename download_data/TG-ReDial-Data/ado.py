import os
import torch
import torch.nn as nn
import torchaudio
import librosa
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from tqdm import tqdm

device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled = False

def load_audio(file_path, target_sample_rate=16000, max_length=None):
    try:
        waveform, sample_rate = librosa.load(
            file_path, 
            sr=target_sample_rate, 
            mono=True,
            res_type='kaiser_best' 
        )
        waveform = torch.from_numpy(waveform)
        if max_length is not None:
            if len(waveform) > max_length:
                waveform = waveform[:max_length]
            else:
                waveform = torch.nn.functional.pad(waveform, (0, max_length - len(waveform)))
        
        return waveform
    except Exception as e:
        print(f"加载音频失败: {e}")
        return None

class AudioEmbeddingExtractor(nn.Module):
    def __init__(self, model_name="facebook/wav2vec2-base", output_dim=1024):
        super().__init__()
        self.model = Wav2Vec2Model.from_pretrained(model_name)
        hidden_size = self.model.config.hidden_size
        self.projection = nn.Linear(hidden_size, output_dim)
        self.activation = nn.ReLU()
        
    def forward(self, inputs):
        outputs = self.model(**inputs)
        hidden_states = outputs.last_hidden_state
        pooled = torch.mean(hidden_states, dim=1)
        projected = self.projection(pooled)
        return self.activation(projected)

def get_wav2vec_embeddings(audio_files, output_dir, model_name="facebook/wav2vec2-base", output_dim=1024):
    """
    使用Wav2Vec2模型处理音频文件并保存1024维嵌入向量
    
    参数:
        audio_files: 音频文件路径列表
        output_dir: 嵌入向量保存目录
        model_name: Wav2Vec2模型名称
        output_dim: 输出向量维度
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载模型和处理器
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = AudioEmbeddingExtractor(model_name, output_dim).to(device)
    model.eval()
    
    for audio_path in tqdm(audio_files, desc="Processing audio files"):
        # 加载和预处理音频
        base_name = os.path.basename(audio_path).replace(".mp3", ".pt")
        save_path = os.path.join(output_dir, base_name)
        if os.path.exists(save_path):
            print(save_path)
            continue
            
        waveform = load_audio(audio_path)
        if waveform is None:
            continue
        try:
            inputs = processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)
            
            # 移动到设备并获取嵌入
            with torch.no_grad():
                inputs = {k: v.to(device) for k, v in inputs.items()}
                embeddings = model(inputs)
            
            # 保存嵌入向量
            base_name = os.path.basename(audio_path).replace(".mp3", ".pt")
            save_path = os.path.join(output_dir, base_name)
            torch.save(embeddings.cpu().squeeze(0), save_path)
        except Exception as e:
            print(f"加载音频失败: {e}")

if __name__ == "__main__":
    # 配置参数
    audio_dir = "audio/"  # 替换为你的MP3文件夹路径
    output_dir = "audio_emb/"   # 嵌入向量输出目录
    
    # 获取所有MP3文件
    audio_files = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) 
                  if f.endswith(".mp3")]
    
    # 处理音频文件
    get_wav2vec_embeddings(audio_files, output_dir)