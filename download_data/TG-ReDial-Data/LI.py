import os
import pandas as pd
from PIL import Image
from icrawler.builtin import GoogleImageCrawler
import glob

with open('movies_with_mentions.csv', 'r', encoding='GBK', errors='ignore') as f:
    content = f.read()
from io import StringIO
df = pd.read_csv(StringIO(content))
df.columns = df.columns.str.strip()
print(df.columns) 
os.makedirs('image', exist_ok=True)
cookies_path = "cookies.txt"
for i in glob.glob(os.path.join("image_","*")):
    os.remove(i)

def download_google_poster(movie_name, db_id,save_dir="image_"):
    os.makedirs(save_dir, exist_ok=True)
    crawler = GoogleImageCrawler(storage={"root_dir": save_dir})
    crawler.crawl(keyword=f"{movie_name} movie poster", max_num=1)
    try:
        os.rename((glob.glob("image_/*"))[0], os.path.join("image", f"{db_id}_raw.jpg"))
        return
    except:
        pass

def compress_image(input_path, output_path, size=(480, 720)):
    if not os.path.exists(input_path):
        return
    img = Image.open(input_path)
    img = img.convert("RGB")
    img = img.resize(size)
    img.save(output_path)

for _, row in df.iterrows():
    print(_)
    global_id = row['global_id']
    movie_name = row['movieName']
    db_id = row['db_id']
    if os.path.exists(f"image/{db_id}.jpg"):
        if os.path.exists(f"image/{db_id}_raw.jpg"):
            os.remove(f"image/{db_id}_raw.jpg")
        continue
    print(f"处理电影: {movie_name} (db_id: {db_id})")
    if not os.path.exists(f"image/{db_id}_raw.jpg"):
        download_google_poster(movie_name,db_id) 
    compress_image(f"image/{db_id}_raw.jpg", f"image/{db_id}.jpg")
    print("done!")
    if os.path.exists(f"image/{db_id}_raw.jpg"):
        os.remove(f"image/{db_id}_raw.jpg")
    