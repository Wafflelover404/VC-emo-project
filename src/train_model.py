import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, f1_score
from sklearn.preprocessing import label_binarize
import os
import json
import time
import sys


sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.metrics import compute_accuracy, compute_f1, plot_confusion_matrix, plot_roc_curve


train_path = "train"
test_path = "test"
model_save_path = os.environ.get("MODEL_PATH", "wafflelover404_emotion_model.pth")

fast_mode = os.environ.get("FAST", "0") == "1"
img_size = int(os.environ.get("IMG_SIZE", "128" if fast_mode else "224"))
batch_size = int(os.environ.get("BATCH_SIZE", "128" if fast_mode else "32"))
num_epochs = int(os.environ.get("EPOCHS", "2" if fast_mode else "30"))

lr = float(os.environ.get("LR", "0.001" if fast_mode else "0.001"))
weight_decay = float(os.environ.get("WEIGHT_DECAY", "0.0" if fast_mode else "0.001"))
unfreeze = os.environ.get("UNFREEZE", "none" if fast_mode else "all")


classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
num_classes = len(classes)

os.makedirs('logs', exist_ok=True)
log_path = os.path.join('logs', 'train.log')

def log(msg: str) -> None:
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    line = f'[{ts}] {msg}'
    print(line, flush=True)
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


image_datasets = {
    'train': datasets.ImageFolder(train_path, data_transforms['train']),
    'test': datasets.ImageFolder(test_path, data_transforms['test']),
}

class_counts = np.bincount([label for _, label in image_datasets['train'].samples], minlength=num_classes)
class_weights = class_counts.sum() / np.maximum(class_counts, 1)
class_weights = class_weights / class_weights.mean()


dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=0),
    'test': DataLoader(image_datasets['test'], batch_size=batch_size, shuffle=False, num_workers=0),
}


model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)


if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
model = model.to(device)

log(f'Device: {device}')
log(f'FAST={1 if fast_mode else 0} IMG_SIZE={img_size} BATCH_SIZE={batch_size} EPOCHS={num_epochs}')
log(f'LR={lr} WEIGHT_DECAY={weight_decay} UNFREEZE={unfreeze}')
log(f'Train samples: {len(image_datasets["train"])}  Test samples: {len(image_datasets["test"])}')

log('Train class distribution: ' + ', '.join([f'{classes[i]}={int(class_counts[i])}' for i in range(num_classes)]))

for name, param in model.named_parameters():
    param.requires_grad = True

if unfreeze == 'none':
    for name, param in model.named_parameters():
        if not name.startswith('fc'):
            param.requires_grad = False
elif unfreeze == 'layer4':
    for name, param in model.named_parameters():
        if not (name.startswith('layer4') or name.startswith('fc')):
            param.requires_grad = False
elif unfreeze == 'all':
    pass
else:
    log(f'Unknown UNFREEZE="{unfreeze}", falling back to layer4')
    for name, param in model.named_parameters():
        if not (name.startswith('layer4') or name.startswith('fc')):
            param.requires_grad = False


criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device), label_smoothing=0.1)
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)

warmup_epochs = 2
def lr_lambda(epoch):
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (num_epochs - warmup_epochs)))

scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_model(model, criterion, optimizer, num_epochs=10):
    best_test_acc = -1.0
    best_path = os.path.splitext(model_save_path)[0] + '_best.pth'
    for epoch in range(num_epochs):
        epoch_start = time.time()
        log(f'Epoch {epoch+1}/{num_epochs}')
        log(f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            num_batches = len(dataloaders[phase])
            phase_start = time.time()
            last_log_t = phase_start
            for batch_idx, (inputs, labels) in enumerate(dataloaders[phase], start=1):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                now = time.time()
                if (now - last_log_t) >= 10:
                    done = batch_idx
                    elapsed = now - phase_start
                    sec_per_batch = elapsed / max(done, 1)
                    eta_sec = sec_per_batch * (num_batches - done)
                    log(f'  {phase}: batch {done}/{num_batches}  loss={loss.item():.4f}  {sec_per_batch:.2f}s/batch  ETA={eta_sec/60:.1f}m')
                    last_log_t = now

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.float() / len(image_datasets[phase])
            log(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'test':
                test_acc = float(epoch_acc.detach().cpu().item())
                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                    torch.save(model.state_dict(), best_path)
                    log(f'New best test acc: {best_test_acc:.4f} -> saved {best_path}')

        epoch_time = time.time() - epoch_start
        log(f'Epoch time: {epoch_time/60:.2f} min')
        scheduler.step()
    return model


model = train_model(model, criterion, optimizer, num_epochs=num_epochs)


torch.save(model.state_dict(), model_save_path)


model.eval()
all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():
    for inputs, labels in dataloaders['test']:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())


accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
print(f'Test Accuracy: {accuracy:.4f}')

f1 = f1_score(all_labels, all_preds, average='macro')
print(f'Test F1 (macro): {f1:.4f}')


cm = confusion_matrix(all_labels, all_preds)
print('Confusion Matrix:')
print(cm)


report = classification_report(all_labels, all_preds, target_names=classes)
print('Classification Report:')
print(report)


all_labels_bin = label_binarize(all_labels, classes=range(num_classes))
all_probs = np.array(all_probs)

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(all_labels_bin[:, i], all_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])


plt.figure()
for i in range(num_classes):
    plt.plot(fpr[i], tpr[i], label=f'{classes[i]} (AUC = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
os.makedirs('metrics', exist_ok=True)
plt.savefig('metrics/roc_curve.png')
plt.close()


with open('metrics/accuracy.txt', 'w') as f:
    f.write(f'{accuracy:.4f}\n')
with open('metrics/f1_score.txt', 'w') as f:
    f.write(f'{f1:.4f}\n')
np.savetxt('metrics/confusion_matrix.txt', cm, fmt='%d')
with open('metrics/roc_auc.json', 'w') as f:
    json.dump(roc_auc, f)
with open('metrics/classification_report.txt', 'w') as f:
    f.write(report)
