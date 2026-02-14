#!/usr/bin/env python3
import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision import transforms, datasets
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, 
    classification_report, roc_curve, auc, label_binarize
)

CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def load_model(model_path: str, num_classes: int):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state = torch.load(model_path, map_location=device)
    
    if 'efficientnet' in model_path.lower():
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        model = efficientnet_b0(weights=None)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    elif 'mobilenet' in model_path.lower():
        from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
        model = mobilenet_v2(weights=None)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        from torchvision.models import resnet18, ResNet18_Weights
        model = resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return model, device

def find_models(models_dir='models'):
    if not os.path.exists(models_dir):
        return []
    return [f for f in os.listdir(models_dir) if f.endswith('.pth')]

def test_model(model_path, test_path, img_size=224, batch_size=32):
    print(f"Loading model: {model_path}")
    model, device = load_model(model_path, len(CLASSES))
    
    data_transform = transforms.Compose([
        transforms.Resize(img_size + 32),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    ds = datasets.ImageFolder(test_path, data_transform)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"Testing on {len(ds)} images...")
    
    all_labels = []
    all_preds = []
    all_probs = []
    
    model.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy())
    
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)
    
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average='macro')
    cm = confusion_matrix(y_true, y_pred)
    
    print("\n" + "="*50)
    print(f"Model: {os.path.basename(model_path)}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Macro: {f1m:.4f}")
    print("="*50)
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASSES))
    
    print("\nConfusion Matrix:")
    print(cm)
    
    y_true_bin = label_binarize(y_true, classes=list(range(len(CLASSES))))
    roc_auc = {}
    
    fig = plt.figure(figsize=(10, 8))
    for i, cls in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc[cls] = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{cls} (AUC = {roc_auc[cls]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    plt.savefig(f'{model_name}_roc.png', dpi=150, bbox_inches='tight')
    print(f"\nROC curve saved to: {model_name}_roc.png")
    
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(CLASSES))
    plt.xticks(tick_marks, CLASSES, rotation=45)
    plt.yticks(tick_marks, CLASSES)
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(f'{model_name}_confusion.png', dpi=150, bbox_inches='tight')
    print(f"Confusion matrix saved to: {model_name}_confusion.png")
    
    return {
        'accuracy': acc,
        'f1_macro': f1m,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
    }

def main():
    parser = argparse.ArgumentParser(description='Test emotion recognition model')
    parser.add_argument('--model', '-m', type=str, help='Path to model .pth file')
    parser.add_argument('--test', '-t', type=str, default='test', help='Path to test dataset')
    parser.add_argument('--img-size', '-i', type=int, default=224, help='Image size')
    parser.add_argument('--batch-size', '-b', type=int, default=32, help='Batch size')
    args = parser.parse_args()
    
    models_dir = 'models'
    available_models = find_models(models_dir)
    
    if args.model:
        model_path = args.model
    elif available_models:
        print("Available models:")
        for i, m in enumerate(available_models):
            print(f"  [{i+1}] {m}")
        print(f"  [0] Custom path")
        
        choice = input("\nSelect model: ").strip()
        if choice == '0':
            model_path = input("Enter model path: ").strip()
        else:
            idx = int(choice) - 1
            model_path = os.path.join(models_dir, available_models[idx])
    else:
        print("No models found in 'models/' directory.")
        model_path = input("Enter model path: ").strip()
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)
    
    test_path = args.test
    if not os.path.exists(test_path):
        print(f"Error: Test directory not found: {test_path}")
        sys.exit(1)
    
    test_model(model_path, test_path, args.img_size, args.batch_size)

if __name__ == '__main__':
    main()
