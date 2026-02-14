#!/usr/bin/env python3
import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision import transforms, datasets
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, 
    classification_report, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from datetime import datetime

CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def load_model(model_path: str, num_classes: int):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state = torch.load(model_path, map_location=device)
    
    if 'efficientnet' in model_path.lower():
        from torchvision.models import efficientnet_b0
        model = efficientnet_b0(weights=None)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    elif 'mobilenet' in model_path.lower():
        from torchvision.models import mobilenet_v2
        model = mobilenet_v2(weights=None)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        from torchvision.models import resnet18
        model = resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return model, device

def test_model(model_path, test_path, img_size=224, batch_size=32):
    print(f"Загрузка модели: {model_path}")
    model, device = load_model(model_path, len(CLASSES))
    
    data_transform = transforms.Compose([
        transforms.Resize(img_size + 32),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    ds = datasets.ImageFolder(test_path, data_transform)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"Тестирование на {len(ds)} изображениях...")
    
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
    rep = classification_report(y_true, y_pred, target_names=CLASSES, output_dict=True)
    
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
    
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Матрица ошибок')
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
    plt.ylabel('Истинный класс')
    plt.xlabel('Предсказанный класс')
    plt.tight_layout()
    plt.savefig(f'{model_name}_confusion.png', dpi=150, bbox_inches='tight')
    
    return {
        'accuracy': acc,
        'f1_macro': f1m,
        'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
        'roc_auc': roc_auc,
        'confusion_matrix': cm.tolist(),
        'classification_report': rep,
        'total_images': len(ds),
    }

def generate_markdown_report(model_path, test_path, output_path='report.md'):
    results = test_model(model_path, test_path)
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    
    md = f"""# 📊 Отчёт о тестировании модели

**Модель:** `{os.path.basename(model_path)}`  
**Дата:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Тестовый набор:** `{test_path}`  
**Изображений:** {results['total_images']}

---

## 🎯 Основные метрики

| Метрика | Значение |
|---------|----------|
| Accuracy | **{results['accuracy']:.4f}** |
| F1 Macro | **{results['f1_macro']:.4f}** |
| F1 Weighted | **{results['f1_weighted']:.4f}** |
| Avg ROC AUC | **{np.mean(list(results['roc_auc'].values())):.4f}** |

---

## 📈 ROC AUC по классам

| Класс | ROC AUC |
|-------|---------|
"""
    
    for cls, auc_val in results['roc_auc'].items():
        md += f"| {cls} | {auc_val:.4f} |\n"
    
    md += f"""
---

## 🎯 Матрица ошибок

![Матрица ошибок]({model_name}_confusion.png)

---

## 📊 ROC Curve

![ROC Curve]({model_name}_roc.png)

---

## 📋 Classification Report

| Класс | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
"""
    
    for cls in CLASSES:
        r = results['classification_report'][cls]
        md += f"| {cls} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1-score']:.4f} | {int(r['support'])} |\n"
    
    md += f"""
---

## 📊 Confusion Matrix (числа)

```
{results['confusion_matrix']}
```

---

*Отчёт сгенерирован автоматически*
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md)
    
    print(f"\n✅ Отчёт сохранён в: {output_path}")
    print(f"✅ ROC кривая: {model_name}_roc.png")
    print(f"✅ Матрица ошибок: {model_name}_confusion.png")

if __name__ == '__main__':
    model_path = 'models/wafflelover404_max.pth'
    test_path = 'test'
    
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    if len(sys.argv) > 2:
        test_path = sys.argv[2]
    
    generate_markdown_report(model_path, test_path)
