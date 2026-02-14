import os
import shutil
import random

train_dir = 'train'
test_dir = 'test'
split_ratio = 0.2  # 20% to test

# Create test subdirs if not exist
for emotion in ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']:
    os.makedirs(os.path.join(test_dir, emotion), exist_ok=True)

# Split for each emotion
for emotion in ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']:
    train_emotion_dir = os.path.join(train_dir, emotion)
    test_emotion_dir = os.path.join(test_dir, emotion)

    if not os.path.exists(train_emotion_dir):
        continue

    files = os.listdir(train_emotion_dir)
    random.shuffle(files)
    split_index = int(len(files) * split_ratio)

    test_files = files[:split_index]

    for file in test_files:
        shutil.move(os.path.join(train_emotion_dir, file), os.path.join(test_emotion_dir, file))

print("Dataset split into train and test.")
