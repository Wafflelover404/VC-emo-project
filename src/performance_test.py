import os
import cv2
import torch
import time
import numpy as np
from torchvision import transforms, models
import torch.nn as nn

# Параметры
model_path = os.environ.get('MODEL_PATH', 'wafflelover404_emotion_model.pth')
img_size = int(os.environ.get('IMG_SIZE', '128'))
classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
num_classes = len(classes)

# Устройство
if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

def load_model(model_path: str, num_classes: int) -> torch.nn.Module:
    # Load the state dict first to detect architecture
    state = torch.load(model_path, map_location=device)
    
    # Detect model architecture from state dict keys
    if 'conv1.weight' in state:
        # ResNet architecture
        model = models.resnet18()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif 'features.0.0.weight' in state:
        # EfficientNet or MobileNet architecture - check filename
        if 'efficientnet' in model_path.lower():
            from torchvision.models import efficientnet_b0
            model = efficientnet_b0()
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        elif 'mobilenet' in model_path.lower():
            from torchvision.models import mobilenet_v2
            model = mobilenet_v2()
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        else:
            # Default to EfficientNet if unsure
            from torchvision.models import efficientnet_b0
            model = efficientnet_b0()
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        # Fallback to ResNet
        model = models.resnet18()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return model

# Загрузка модели
model = load_model(model_path, num_classes)

# Трансформации
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(img_size + 32),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Детекция лиц
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def benchmark_inference():
    """Тестирование скорости инференса"""
    print("🚀 Тестирование производительности инференса...")
    
    # Создаем тестовое изображение (48x48 как в FER2013)
    test_image = np.random.randint(0, 255, (48, 48, 3), dtype=np.uint8)
    
    # Прогрев модели
    for _ in range(5):
        input_tensor = transform(test_image).unsqueeze(0).to(device)
        with torch.no_grad():
            _ = model(input_tensor)
    
    # Измерение времени
    times = []
    num_iterations = 100
    
    for i in range(num_iterations):
        start_time = time.time()
        
        # Предобработка
        input_tensor = transform(test_image).unsqueeze(0).to(device)
        
        # Инференс
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, pred = torch.max(probs, 1)
        
        end_time = time.time()
        iteration_time = end_time - start_time
        times.append(iteration_time)
        
        if i < 5:  # Показываем первые несколько результатов
            emotion = classes[pred.item()]
            conf = confidence.item()
            print(f"  Итерация {i+1}: {emotion} (confidence: {conf:.3f}) - {iteration_time:.4f}s")
    
    avg_time = np.mean(times)
    min_time = np.min(times)
    max_time = np.max(times)
    fps = 1.0 / avg_time
    
    print(f"\n📊 Результаты производительности ({num_iterations} итераций):")
    print(f"  Среднее время: {avg_time:.4f}s")
    print(f"  Минимальное: {min_time:.4f}s")
    print(f"  Максимальное: {max_time:.4f}s")
    print(f"  Потенциальный FPS: {fps:.1f}")
    
    # Проверка требования ≤2с
    if avg_time <= 2.0:
        print(f"✅ Требование выполнено: среднее время {avg_time:.4f}s ≤ 2s")
    else:
        print(f"❌ Требование не выполнено: среднее время {avg_time:.4f}s > 2s")
    
    return avg_time <= 2.0

def benchmark_full_pipeline():
    """Тестирование полного конвейера с детекцией лиц"""
    print("\n🎯 Тестирование полного конвейера (детекция + инференс)...")
    
    # Создаем тестовый кадр большего размера
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    times = []
    num_iterations = 50
    
    for i in range(num_iterations):
        start_time = time.time()
        
        # Детекция лиц
        gray = cv2.cvtColor(test_frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # Инференс для каждого лица
        for (x, y, w, h) in faces:
            face_roi = test_frame[y:y+h, x:x+w]
            if face_roi.size > 0:
                input_tensor = transform(face_roi).unsqueeze(0).to(device)
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    confidence, pred = torch.max(probs, 1)
        
        end_time = time.time()
        iteration_time = end_time - start_time
        times.append(iteration_time)
    
    avg_time = np.mean(times)
    fps = 1.0 / avg_time
    
    print(f"  Полный конвейер - среднее время: {avg_time:.4f}s")
    print(f"  Потенциальный FPS с детекцией: {fps:.1f}")
    
    return avg_time <= 2.0

if __name__ == "__main__":
    print("=== Тестирование производительности ===")
    print(f"Устройство: {device}")
    print(f"Размер изображения: {img_size}x{img_size}")
    print(f"Модель: {model_path}")
    print()
    
    # Тест инференса
    inference_ok = benchmark_inference()
    
    # Тест полного конвейера
    pipeline_ok = benchmark_full_pipeline()
    
    print(f"\n🎯 Итоги:")
    print(f"  Инференс: {'✅' if inference_ok else '❌'}")
    print(f"  Полный конвейер: {'✅' if pipeline_ok else '❌'}")
    
    if inference_ok and pipeline_ok:
        print("🚀 Все требования производительности выполнены!")
    else:
        print("⚠️ Требуются оптимизации для соответствия требованиям")
