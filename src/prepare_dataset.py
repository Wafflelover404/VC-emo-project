import pandas as pd
import numpy as np
from PIL import Image
import os


csv_path = 'fer2013.csv'  
train_dir = 'train'
test_dir = 'test'


for dir_path in [train_dir, test_dir]:
    for emotion in ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']:
        os.makedirs(os.path.join(dir_path, emotion), exist_ok=True)


emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}


df = pd.read_csv(csv_path)


for index, row in df.iterrows():
    emotion = emotion_labels[row['emotion']]
    pixels = np.array(row['pixels'].split(), dtype=np.uint8).reshape(48, 48)
    img = Image.fromarray(pixels, mode='L')  

    
    if row['Usage'] == 'Training':
        save_dir = train_dir
    else:  
        save_dir = test_dir

    
    img_path = os.path.join(save_dir, emotion, f'{index}.png')
    img.save(img_path)

print("Dataset prepared.")
