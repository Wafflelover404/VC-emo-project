#!/usr/bin/env python3
"""
Console application for emotion recognition from file.
Usage: python file_inference.py path/to/image.jpg
"""

import sys
import os
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn

# Configuration
MODEL_PATH = os.environ.get('MODEL_PATH', 'wafflelover404_emotion_model.pth')
IMG_SIZE = int(os.environ.get('IMG_SIZE', '128'))

# Classes
classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
num_classes = len(classes)

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def load_model(model_path: str, num_classes: int) -> Tuple[torch.nn.Module, torch.device]:
    device = get_device()
    
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
    print(f"✓ Model loaded from {model_path}")
    return model, device

def preprocess_image(image_path, img_size):
    """Load and preprocess image"""
    try:
        # Load image
        if not os.path.exists(image_path):
            print(f"✗ File not found: {image_path}")
            return None
            
        # Read with PIL first
        image = Image.open(image_path).convert('RGB')
        print(f"✓ Image loaded: {image.size}, mode: {image.mode}")
        
        # Convert to numpy array (BGR for OpenCV)
        image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Preprocess for model
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(img_size + 32),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        image_tensor = transform(image_bgr).unsqueeze(0)
        return image_tensor, image_bgr
        
    except Exception as e:
        print(f"✗ Error processing image: {e}")
        return None

def detect_faces(image_bgr):
    """Detect faces in image using Haar Cascade"""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

def predict_emotion(model, image_tensor, device):
    """Predict emotion from image tensor"""
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, 1)
        emotion = classes[pred.item()]
        conf = confidence.item()
        return emotion, conf, probs.squeeze(0).cpu().numpy()

def main():
    if len(sys.argv) != 2:
        print("Usage: python file_inference.py <path_to_image>")
        print("Example: python file_inference.py test/happy/PrivateTest_10077120.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    print(f"Emotion Recognition from File")
    print(f"Image: {image_path}")
    print(f"Model: {MODEL_PATH}")
    print(f"Image size: {IMG_SIZE}x{IMG_SIZE}")
    print("-" * 40)
    
    # Setup
    device = get_device()
    print(f"Device: {device}")
    
    # Load model
    model = load_model(MODEL_PATH, device)
    
    # Process image
    result = preprocess_image(image_path, IMG_SIZE)
    if result is None:
        sys.exit(1)
    
    image_tensor, image_bgr = result
    
    # Detect faces
    faces = detect_faces(image_bgr)
    print(f"Faces detected: {len(faces)}")
    
    if len(faces) > 0:
        print("\n--- Face Detection Results ---")
        for i, (x, y, w, h) in enumerate(faces):
            print(f"Face {i+1}: position=({x},{y}), size=({w}x{h})")
            
            # Extract face ROI
            face_roi = image_bgr[y:y+h, x:x+w]
            
            # Preprocess face
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(IMG_SIZE + 32),
                transforms.CenterCrop(IMG_SIZE),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            face_tensor = transform(face_roi).unsqueeze(0)
            
            # Predict emotion
            emotion, conf, probs = predict_emotion(model, face_tensor, device)
            
            print(f"  Emotion: {emotion}")
            print(f"  Confidence: {conf:.2%}")
            print(f"  All probabilities:")
            for cls, prob in zip(classes, probs):
                print(f"    {cls}: {prob:.2%}")
            print()
    else:
        # Use whole image
        print("\n--- Whole Image Analysis ---")
        emotion, conf, probs = predict_emotion(model, image_tensor, device)
        
        print(f"Emotion: {emotion}")
        print(f"Confidence: {conf:.2%}")
        print(f"All probabilities:")
        for cls, prob in zip(classes, probs):
            print(f"  {cls}: {prob:.2%}")

if __name__ == "__main__":
    main()
