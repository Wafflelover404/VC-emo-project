#!/usr/bin/env python3
"""
Queue-based training script for emotion detection models.
Runs multiple training configurations sequentially and saves models with metadata.
"""

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
import datetime
from typing import Dict, List, Any
import copy

# Add utils path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.metrics import compute_accuracy, compute_f1, plot_confusion_matrix, plot_roc_curve

# Configuration queue - different model architectures and hyperparameters
TRAINING_QUEUE = [
    {
        "name": "resnet18_baseline",
        "model_type": "resnet18",
        "img_size": 224,
        "batch_size": 32,
        "epochs": 30,
        "lr": 0.0003,
        "weight_decay": 0.01,
        "unfreeze": "layer4",
        "description": "Baseline ResNet18 with current hyperparameters"
    }
]

class ModelTrainer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.setup_paths()
        self.setup_device()
        self.setup_data()
        self.setup_model()
        
    def setup_paths(self):
        """Setup paths for logging and model saving"""
        self.models_dir = "models"
        self.logs_dir = "logs"
        self.metrics_dir = "metrics"
        
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)
        
        # Create unique model filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_name = f"{self.config['name']}_{timestamp}"
        self.model_path = os.path.join(self.models_dir, f"{self.model_name}.pth")
        self.best_model_path = os.path.join(self.models_dir, f"{self.model_name}_best.pth")
        self.metadata_path = os.path.join(self.models_dir, f"{self.model_name}_metadata.json")
        
        # Setup logging
        self.log_path = os.path.join(self.logs_dir, f"{self.model_name}.log")
        
    def setup_device(self):
        """Setup training device"""
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
            
    def setup_data(self):
        """Setup data loaders with configuration-specific transforms"""
        train_path = "train"
        test_path = "test"
        
        img_size = self.config['img_size']
        batch_size = self.config['batch_size']
        
        self.data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(img_size + 32),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        
        self.image_datasets = {
            'train': datasets.ImageFolder(train_path, self.data_transforms['train']),
            'test': datasets.ImageFolder(test_path, self.data_transforms['test']),
        }
        
        self.classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        self.num_classes = len(self.classes)
        
        # Calculate class weights
        class_counts = np.bincount([label for _, label in self.image_datasets['train'].samples], 
                                  minlength=self.num_classes)
        class_weights = class_counts.sum() / np.maximum(class_counts, 1)
        class_weights = class_weights / class_weights.mean()
        
        self.dataloaders = {
            'train': DataLoader(self.image_datasets['train'], batch_size=batch_size, 
                              shuffle=True, num_workers=0),
            'test': DataLoader(self.image_datasets['test'], batch_size=batch_size, 
                             shuffle=False, num_workers=0),
        }
        
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32, device=self.device)
        
    def setup_model(self):
        """Setup model architecture based on configuration"""
        model_type = self.config['model_type']
        
        if model_type == 'resnet18':
            self.model = models.resnet18(pretrained=True)
            self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)
        elif model_type == 'resnet34':
            self.model = models.resnet34(pretrained=True)
            self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)
        elif model_type == 'mobilenet_v2':
            self.model = models.mobilenet_v2(pretrained=True)
            self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, self.num_classes)
        elif model_type == 'efficientnet_b0':
            self.model = models.efficientnet_b0(pretrained=True)
            self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, self.num_classes)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        self.model = self.model.to(self.device)
        
        # Setup parameter freezing based on unfreeze strategy
        unfreeze = self.config.get('unfreeze', 'layer4')
        for name, param in self.model.named_parameters():
            param.requires_grad = True
            
        if unfreeze == 'none':
            for name, param in self.model.named_parameters():
                if not name.startswith(('fc', 'classifier')):
                    param.requires_grad = False
        elif unfreeze == 'layer4':
            for name, param in self.model.named_parameters():
                if not (name.startswith('layer4') or name.startswith(('fc', 'classifier'))):
                    param.requires_grad = False
        # 'all' keeps all parameters trainable
        
    def log(self, msg: str) -> None:
        """Logging function"""
        ts = time.strftime('%Y-%m-%d %H:%M:%S')
        line = f'[{ts}] {msg}'
        print(line, flush=True)
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(line + '\n')
            
    def train_model(self):
        """Main training loop"""
        self.log(f'Starting training: {self.config["name"]}')
        self.log(f'Description: {self.config["description"]}')
        self.log(f'Device: {self.device}')
        self.log(f'Model: {self.config["model_type"]}')
        self.log(f'IMG_SIZE={self.config["img_size"]} BATCH_SIZE={self.config["batch_size"]} EPOCHS={self.config["epochs"]}')
        self.log(f'LR={self.config["lr"]} WEIGHT_DECAY={self.config["weight_decay"]} UNFREEZE={self.config["unfreeze"]}')
        self.log(f'Train samples: {len(self.image_datasets["train"])}  Test samples: {len(self.image_datasets["test"])}')
        
        criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), 
                              lr=self.config['lr'], weight_decay=self.config['weight_decay'])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(self.config['epochs'], 1))
        
        best_test_acc = -1.0
        training_history = {
            'train_loss': [], 'train_acc': [],
            'test_loss': [], 'test_acc': [],
            'learning_rates': []
        }
        
        for epoch in range(self.config['epochs']):
            epoch_start = time.time()
            self.log(f'Epoch {epoch+1}/{self.config["epochs"]}')
            self.log(f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
            
            for phase in ['train', 'test']:
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()
                    
                running_loss = 0.0
                running_corrects = 0
                num_batches = len(self.dataloaders[phase])
                phase_start = time.time()
                last_log_t = phase_start
                
                for batch_idx, (inputs, labels) in enumerate(self.dataloaders[phase], start=1):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                            
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    
                    # Progress logging every 10 seconds
                    now = time.time()
                    if (now - last_log_t) >= 10:
                        done = batch_idx
                        elapsed = now - phase_start
                        sec_per_batch = elapsed / max(done, 1)
                        eta_sec = sec_per_batch * (num_batches - done)
                        self.log(f'  {phase}: batch {done}/{num_batches}  loss={loss.item():.4f}  {sec_per_batch:.2f}s/batch  ETA={eta_sec/60:.1f}m')
                        last_log_t = now
                        
                epoch_loss = running_loss / len(self.image_datasets[phase])
                epoch_acc = running_corrects.float() / len(self.image_datasets[phase])
                
                training_history[f'{phase}_loss'].append(epoch_loss)
                training_history[f'{phase}_acc'].append(float(epoch_acc.detach().cpu().item()))
                
                self.log(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                
                if phase == 'test':
                    test_acc = float(epoch_acc.detach().cpu().item())
                    if test_acc > best_test_acc:
                        best_test_acc = test_acc
                        torch.save(self.model.state_dict(), self.best_model_path)
                        self.log(f'New best test acc: {best_test_acc:.4f} -> saved {self.best_model_path}')
                        
            training_history['learning_rates'].append(optimizer.param_groups[0]['lr'])
            epoch_time = time.time() - epoch_start
            self.log(f'Epoch time: {epoch_time/60:.2f} min')
            scheduler.step()
            
        return training_history, best_test_acc
        
    def evaluate_model(self):
        """Comprehensive model evaluation"""
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for inputs, labels in self.dataloaders['test']:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                
        # Calculate metrics
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        f1 = f1_score(all_labels, all_preds, average='macro')
        cm = confusion_matrix(all_labels, all_preds)
        report = classification_report(all_labels, all_preds, target_names=self.classes)
        
        # ROC curves
        all_labels_bin = label_binarize(all_labels, classes=range(self.num_classes))
        all_probs = np.array(all_probs)
        
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(self.num_classes):
            fpr[i], tpr[i], _ = roc_curve(all_labels_bin[:, i], all_probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            
        return {
            'accuracy': accuracy,
            'f1_macro': f1,
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'roc_auc': {self.classes[i]: float(roc_auc[i]) for i in range(self.num_classes)},
            'predictions': all_preds,
            'labels': all_labels,
            'probabilities': all_probs.tolist()
        }
        
    def save_results(self, training_history: Dict, evaluation_results: Dict, best_test_acc: float):
        """Save model, metadata, and metrics"""
        # Save final model
        torch.save(self.model.state_dict(), self.model_path)
        
        # Save metadata
        metadata = {
            'model_name': self.model_name,
            'config': self.config,
            'training_history': training_history,
            'evaluation_results': {
                'accuracy': evaluation_results['accuracy'],
                'f1_macro': evaluation_results['f1_macro'],
                'best_test_acc': best_test_acc,
                'roc_auc': evaluation_results['roc_auc']
            },
            'model_info': {
                'total_params': sum(p.numel() for p in self.model.parameters()),
                'trainable_params': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                'input_size': self.config['img_size'],
                'num_classes': self.num_classes
            },
            'timestamp': datetime.datetime.now().isoformat(),
            'device': str(self.device)
        }
        
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        # Save metrics to files
        metrics_subdir = os.path.join(self.metrics_dir, self.model_name)
        os.makedirs(metrics_subdir, exist_ok=True)
        
        with open(os.path.join(metrics_subdir, 'accuracy.txt'), 'w') as f:
            f.write(f"{evaluation_results['accuracy']:.4f}\n")
        with open(os.path.join(metrics_subdir, 'f1_score.txt'), 'w') as f:
            f.write(f"{evaluation_results['f1_macro']:.4f}\n")
        np.savetxt(os.path.join(metrics_subdir, 'confusion_matrix.txt'), 
                  evaluation_results['confusion_matrix'], fmt='%d')
        with open(os.path.join(metrics_subdir, 'classification_report.txt'), 'w') as f:
            f.write(evaluation_results['classification_report'])
        with open(os.path.join(metrics_subdir, 'roc_auc.json'), 'w') as f:
            json.dump(evaluation_results['roc_auc'], f, indent=2)
            
        # Plot ROC curve
        plt.figure(figsize=(10, 8))
        for i in range(self.num_classes):
            plt.plot(evaluation_results['roc_curves'][i][0], 
                    evaluation_results['roc_curves'][i][1], 
                    label=f'{self.classes[i]} (AUC = {evaluation_results["roc_auc"][self.classes[i]]:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {self.model_name}')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(metrics_subdir, 'roc_curve.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        self.log(f"Results saved to {self.model_path}")
        self.log(f"Metadata saved to {self.metadata_path}")
        self.log(f"Metrics saved to {metrics_subdir}")

def run_training_queue():
    """Run all training configurations in queue"""
    print("Starting training queue...")
    print(f"Total configurations to train: {len(TRAINING_QUEUE)}")
    
    results_summary = []
    
    for i, config in enumerate(TRAINING_QUEUE, 1):
        print(f"\n{'='*60}")
        print(f"Training {i}/{len(TRAINING_QUEUE)}: {config['name']}")
        print(f"{'='*60}")
        
        try:
            trainer = ModelTrainer(config)
            training_history, best_test_acc = trainer.train_model()
            evaluation_results = trainer.evaluate_model()
            
            # Add ROC curves to evaluation results
            all_labels_bin = label_binarize(evaluation_results['labels'], 
                                          classes=range(trainer.num_classes))
            all_probs = np.array(evaluation_results['probabilities'])
            
            fpr = dict()
            tpr = dict()
            for j in range(trainer.num_classes):
                fpr[j], tpr[j], _ = roc_curve(all_labels_bin[:, j], all_probs[:, j])
            
            evaluation_results['roc_curves'] = [
                (fpr[j].tolist(), tpr[j].tolist()) for j in range(trainer.num_classes)
            ]
            
            trainer.save_results(training_history, evaluation_results, best_test_acc)
            
            # Summary for this training
            summary = {
                'model_name': trainer.model_name,
                'config': config,
                'final_accuracy': evaluation_results['accuracy'],
                'final_f1': evaluation_results['f1_macro'],
                'best_test_acc': best_test_acc,
                'status': 'completed'
            }
            results_summary.append(summary)
            
            print(f"✅ Completed: {trainer.model_name}")
            print(f"   Accuracy: {evaluation_results['accuracy']:.4f}")
            print(f"   F1-Score: {evaluation_results['f1_macro']:.4f}")
            print(f"   Best Test Acc: {best_test_acc:.4f}")
            
        except Exception as e:
            print(f"❌ Failed to train {config['name']}: {str(e)}")
            summary = {
                'model_name': f"{config['name']}_failed",
                'config': config,
                'error': str(e),
                'status': 'failed'
            }
            results_summary.append(summary)
            
    # Save summary of all trainings
    summary_path = os.path.join("models", f"training_summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
        
    print(f"\n{'='*60}")
    print("Training queue completed!")
    print(f"Summary saved to: {summary_path}")
    
    # Print comparison table
    print(f"\n{'='*60}")
    print("RESULTS COMPARISON")
    print(f"{'='*60}")
    print(f"{'Model':<20} {'Accuracy':<10} {'F1-Score':<10} {'Best Acc':<10} {'Status':<10}")
    print("-" * 70)
    
    for result in results_summary:
        if result['status'] == 'completed':
            print(f"{result['model_name']:<20} {result['final_accuracy']:<10.4f} "
                  f"{result['final_f1']:<10.4f} {result['best_test_acc']:<10.4f} "
                  f"{result['status']:<10}")
        else:
            print(f"{result['model_name']:<20} {'N/A':<10} {'N/A':<10} {'N/A':<10} {result['status']:<10}")

if __name__ == "__main__":
    run_training_queue()
