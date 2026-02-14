import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, f1_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns

def compute_accuracy(y_true, y_pred):
    return np.mean(np.array(y_pred) == np.array(y_true))

def compute_f1(y_true, y_pred, average='macro'):
    return f1_score(y_true, y_pred, average=average)

def plot_confusion_matrix(y_true, y_pred, classes, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    if save_path:
        plt.savefig(save_path)
    plt.show()
    return cm

def plot_roc_curve(y_true, y_prob, classes, save_path=None):
    y_true_bin = label_binarize(y_true, classes=range(len(classes)))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(classes)):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure()
    for i in range(len(classes)):
        plt.plot(fpr[i], tpr[i], label=f'{classes[i]} (AUC = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    if save_path:
        plt.savefig(save_path)
    plt.show()
    return roc_auc

def collect_metrics(y_true, y_pred, y_prob, classes):
    acc = compute_accuracy(y_true, y_pred)
    f1 = compute_f1(y_true, y_pred)
    cm = plot_confusion_matrix(y_true, y_pred, classes)
    roc_auc = plot_roc_curve(y_true, y_prob, classes)
    report = classification_report(y_true, y_pred, target_names=classes)
    return {
        'accuracy': acc,
        'f1_score': f1,
        'confusion_matrix': cm,
        'roc_auc': roc_auc,
        'classification_report': report
    }
