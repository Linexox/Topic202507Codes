import os
import subprocess
import pandas as pd
import requests
import time
import logging
import ffmpeg
import random
from yt_dlp import YoutubeDL
import logging

api_url = "https://dps.kdlapi.com/api/getdps/?secret_id=ozlz0ii7udw2ao9r35e1&signature=7i57uk20tguilkks19wdqd2w4bo55hqs&num=1&pt=1&sep=1"

with open('movies_with_mentions.csv', 'r', encoding='GBK', errors='ignore') as f:
    content = f.read()

from io import StringIO
df = pd.read_csv(StringIO(content))
df.columns = df.columns.str.strip()  # 去除列名中的前后空格
print(df.columns)  # 确认列名

# 存储海报图片的文件夹
os.makedirs('video', exist_ok=True)
os.makedirs('audio', exist_ok=True)

cookies_path = "/root/cookies.txt"

options = {
    "cookiefile": cookies_path,
    "quiet": True,
    "format": "worst",
    "noplaylist": True, 
}

def get_raw(db_id):
    print(db_id)
    movie_name = df[df['db_id'] == db_id]['movieName'].values[0]
    raw_video_path = f"video/{db_id}_raw.mp4"
    options["outtmpl"]=raw_video_path
    with YoutubeDL(options) as ydl:
        print(db_id)
        print(".......")
        try:
            search_results = ydl.extract_info(f"ytsearch:{movie_name} movie trailer", download=False)
        except:
            return 
        if "entries" in search_results and search_results["entries"]:
            try:
                first_video = search_results["entries"][0]
                video_url = first_video["url"] 
                ydl.download([video_url])
            except:
                return
        else:
            print("未找到匹配的视频")
        return


def get_video(db_id):
    raw_video_path = f"video/{db_id}_raw.mp4"
    video_path = f"video/{db_id}.mp4"
    
    if not os.path.exists(raw_video_path):
        logging.error(f"原始视频文件 {raw_video_path} 不存在")
        return
    
    try:
        probe = ffmpeg.probe(raw_video_path)
        total_duration = float(probe["format"]["duration"])
        start_time = max(0, (total_duration - 10) / 2)
        ffmpeg.input(raw_video_path, ss=start_time, t=10).output(
            video_path, vcodec="libx264", crf=28, video_bitrate="1000k", vf="scale=540:720"
        ).run(overwrite_output=True)
        os.remove(raw_video_path)
        logging.info(f"成功处理视频 {video_path}")
        
    except Exception as e:
        logging.error(f"视频处理失败: {str(e)}")

def extract_and_compress_audio(db_id):
    video_path=f"video/{db_id}.mp4"
    audio_path = f"audio/{db_id}.mp3"
    try:
        ffmpeg.input(video_path).output(audio_path, format="mp3", audio_bitrate="128k").run(overwrite_output=True)
        return audio_path
    except Exception as e:
        print(f"提取音频失败：{e}")
        return None

for _, row in df.iterrows():
    print(_)
    global_id = row['global_id']
    movie_name = row['movieName']
    db_id = row['db_id']
    if os.path.exists(f"video/{db_id}_raw.mp4"):
        os.remove(f"video/{db_id}_raw.mp4")
    if os.path.exists(f"audio/{db_id}.mp3"):
        continue
    #if not os.path.exists(f"video/{db_id}.mp4") and not os.path.exists(f"video/{db_id}_raw.mp4"):
     #   get_raw(db_id) 
    #if not os.path.exists(f"video/{db_id}.mp4"):
     #   get_video(db_id) 
    extract_and_compress_audio(db_id) 
    print(f"处理电影: {movie_name} (db_id: {db_id})")
