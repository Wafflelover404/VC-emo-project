import os
import cv2
import torch
from torchvision import transforms
from torchvision import models
import torch.nn as nn
import sys
from PIL import Image
import numpy as np


model_path = os.environ.get('MODEL_PATH', 'wafflelover404_emotion_model.pth')
img_size = int(os.environ.get('IMG_SIZE', '128'))


classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
num_classes = len(classes)


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


model = load_model(model_path, num_classes)


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(img_size + 32),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def process_image_file(image_path):
    """Process a single image file and display results"""
    if not os.path.exists(image_path):
        print(f"Error: File not found: {image_path}")
        return
    
    try:
        # Load image
        image = Image.open(image_path).convert('RGB')
        image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(image_bgr, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) > 0:
            print(f"Found {len(faces)} face(s) in the image")
            for i, (x, y, w, h) in enumerate(faces):
                face_roi = image_bgr[y:y+h, x:x+w]
                face_tensor = transform(face_roi).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    outputs = model(face_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    confidence, pred = torch.max(probs, 1)
                    emotion = classes[pred.item()]
                    conf = confidence.item()
                
                print(f"Face {i+1}: {emotion} (confidence: {conf:.2%})")
                
                # Draw on image
                cv2.rectangle(image_bgr, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(image_bgr, f'{emotion}: {conf:.2f}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        else:
            print("No faces detected, analyzing whole image")
            image_tensor = transform(image_bgr).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(image_tensor)
                probs = torch.softmax(outputs, dim=1)
                confidence, pred = torch.max(probs, 1)
                emotion = classes[pred.item()]
                conf = confidence.item()
            
            print(f"Emotion: {emotion} (confidence: {conf:.2%})")
        
        # Display result
        cv2.imshow('Emotion Detection Result', image_bgr)
        print("Press any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Error processing image: {e}")

def detect_emotion():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    print("Press 'q' to quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            face_tensor = transform(face_roi).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(face_tensor)
                probs = torch.softmax(outputs, dim=1)
                confidence, pred = torch.max(probs, 1)
                emotion = classes[pred.item()]
                conf = confidence.item()

            
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, f'{emotion}: {conf:.2f}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        cv2.imshow('Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # File mode: python camera_inference.py path/to/image.jpg
        image_path = sys.argv[1]
        print(f"Processing image: {image_path}")
        process_image_file(image_path)
    else:
        # Camera mode: python camera_inference.py
        print("Starting camera mode...")
        detect_emotion()
