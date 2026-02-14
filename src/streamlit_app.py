import os
import json
import zipfile
import tempfile
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import auc, classification_report, confusion_matrix, f1_score, roc_curve
from sklearn.preprocessing import label_binarize
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt

CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

@st.cache_resource
def load_model(model_path: str, num_classes: int) -> Tuple[torch.nn.Module, torch.device]:
    device = _get_device()

    state = torch.load(model_path, map_location=device)

    if 'conv1.weight' in state:
        model = models.resnet18()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif 'features.0.0.weight' in state:
        if 'efficientnet' in model_path.lower():
            from torchvision.models import efficientnet_b0
            model = efficientnet_b0()
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        elif 'mobilenet' in model_path.lower():
            from torchvision.models import mobilenet_v2
            model = mobilenet_v2()
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        else:
            from torchvision.models import efficientnet_b0
            model = efficientnet_b0()
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        model = models.resnet18()
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return model, device

@st.cache_resource
def get_face_cascade() -> cv2.CascadeClassifier:
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def get_transform(img_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(img_size + 32),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

def rgb_to_bgr(image_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

def predict_on_bgr_image(
    image_bgr: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    img_size: int,
) -> np.ndarray:
    tfm = get_transform(img_size)
    x = tfm(image_bgr).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()
    return probs

def detect_faces(image_bgr: np.ndarray, scale_factor: float, min_neighbors: int) -> List[Tuple[int, int, int, int]]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = get_face_cascade().detectMultiScale(
        gray, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=(30, 30)
    )
    return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]

def overlay_prediction(
    frame_bgr: np.ndarray,
    faces: List[Tuple[int, int, int, int]],
    probs_list: List[np.ndarray],
) -> np.ndarray:
    out = frame_bgr.copy()
    for (x, y, w, h), probs in zip(faces, probs_list):
        pred_idx = int(np.argmax(probs))
        label = CLASSES[pred_idx]
        conf = float(probs[pred_idx])
        cv2.rectangle(out, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(
            out,
            f'{label}: {conf:.2f}',
            (x, max(0, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 0, 0),
            2,
        )
    return out

def probs_to_table(probs: np.ndarray) -> Dict[str, float]:
    return {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}

def get_available_models():
    models_dir = "models"
    if not os.path.exists(models_dir):
        return []

    model_files = []
    for file in os.listdir(models_dir):
        if file.endswith('.pth'):
            model_files.append(os.path.join(models_dir, file))
    return model_files

def get_available_test_sets():
    test_sets = {}

    if os.path.exists("test") and os.path.isdir("test"):
        test_sets["Стандартный test"] = "test"

    for item in os.listdir("."):
        if os.path.isdir(item) and item != "test" and not item.startswith('.'):
            has_emotion_dirs = False
            for emotion in CLASSES:
                if os.path.exists(os.path.join(item, emotion)):
                    has_emotion_dirs = True
                    break
            if has_emotion_dirs:
                test_sets[item] = item

    return test_sets

def run_model_tests(model, model_path, test_path, device, img_size, batch_size):
    start_time = pd.Timestamp.now()

    with st.spinner(f'🔄 Тестирую модель {os.path.basename(model_path)}...'):
        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text('📂 Подготовка тестовых данных...')
        progress_bar.progress(0.2)

        data_transform = transforms.Compose([
            transforms.Resize(img_size + 32),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        ds = datasets.ImageFolder(test_path, data_transform)
        loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

        total_images = len(ds)
        expected_batches = (total_images + batch_size - 1) // batch_size
        status_text.text(f'📊 Найдено изображений: {total_images} (ожидается батчей: {expected_batches})')
        progress_bar.progress(0.4)

        status_text.text('🧠 Выполнение предсказаний...')
        progress_bar.progress(0.6)

        all_labels: List[int] = []
        all_preds: List[int] = []
        all_probs: List[np.ndarray] = []

        model.eval()
        batch_count = 0

        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                logits = model(inputs)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)

                all_labels.extend(labels.detach().cpu().numpy().tolist())
                all_preds.extend(preds.detach().cpu().numpy().tolist())
                all_probs.extend(probs.detach().cpu().numpy())

                batch_count += 1
                progress = 0.6 + (batch_count / len(loader)) * 0.3
                progress_bar.progress(progress)
                status_text.text(f'🧠 Обработано {batch_count}/{len(loader)} батчей...')

        status_text.text('📈 Вычисление метрик...')
        progress_bar.progress(0.9)

        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        y_prob = np.array(all_probs)

        acc = float(np.mean(y_true == y_pred))
        f1m = float(f1_score(y_true, y_pred, average='macro'))
        cm = confusion_matrix(y_true, y_pred)
        rep = classification_report(y_true, y_pred, target_names=CLASSES)

        y_true_bin = label_binarize(y_true, classes=list(range(len(CLASSES))))
        roc_auc: Dict[str, float] = {}

        fig = plt.figure(figsize=(10, 8))
        for i, cls in enumerate(CLASSES):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            roc_auc[cls] = float(auc(fpr, tpr))
            plt.plot(fpr, tpr, label=f'{cls} (AUC = {roc_auc[cls]:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')

        progress_bar.progress(1.0)

        end_time = pd.Timestamp.now()
        duration = (end_time - start_time).total_seconds()

        status_text.text(f'✅ Тестирование завершено за {duration:.2f} сек')

        test_info = {
            'model_path': model_path,
            'test_path': test_path,
            'img_size': img_size,
            'batch_size': batch_size,
            'total_images': total_images,
            'total_batches': len(loader),
            'duration_seconds': duration,
            'images_per_second': total_images / duration if duration > 0 else 0,
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'device': str(device)
        }

    return TestMetrics(
        accuracy=acc,
        f1_macro=f1m,
        confusion=cm,
        report=rep,
        roc_auc=roc_auc,
        fig_roc=fig,
    ), test_info

def display_model_results(model, model_path, test_path, device, img_size, batch_size, model_name=None):
    try:
        with st.spinner(f'🔄 Тестирую модель {model_name or os.path.basename(model_path)}...'):
            metrics, test_info = run_model_tests(
                model=model,
                model_path=model_path,
                test_path=test_path,
                device=device,
                img_size=img_size,
                batch_size=batch_size
            )

        st.success(f"✅ Тестирование модели {model_name or os.path.basename(model_path)} завершено!")

        with st.expander("📊 Информация о тесте", expanded=True):
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("📁 Файл модели", os.path.basename(test_info['model_path']))
                st.metric("📂 Тестовый набор", os.path.basename(test_info['test_path']))

            with col2:
                st.metric("🖼️ Всего изображений", test_info['total_images'])
                st.metric("📦 Батчей", test_info['total_batches'])

            with col3:
                st.metric("⏱️ Время тестирования", f"{test_info['duration_seconds']:.2f} сек")
                st.metric("🚀 Скорость", f"{test_info['images_per_second']:.1f} img/сек")

            with col4:
                st.metric("📏 Размер изображения", f"{test_info['img_size']}x{test_info['img_size']}")
                st.metric("💾 Batch size", test_info['batch_size'])

        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f'{metrics.accuracy:.4f}')
        col2.metric("F1 Macro", f'{metrics.f1_macro:.4f}')
        col3.metric("Avg ROC AUC", f'{np.mean(list(metrics.roc_auc.values())):.4f}')

        st.markdown("### 📊 ROC AUC по классам")
        roc_df = pd.DataFrame([
            {'Класс': cls, 'ROC AUC': f'{auc:.4f}'}
            for cls, auc in metrics.roc_auc.items()
        ])
        st.dataframe(roc_df, use_container_width=True)

        st.markdown("### 🎯 Confusion Matrix")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(metrics.confusion, annot=True, fmt='d', cmap='Blues',
                   xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig)

        st.markdown("### 📋 Classification Report")
        st.text(metrics.report)

        return metrics, test_info

    except Exception as e:
        st.error(f"❌ Ошибка при тестировании модели: {e}")
        return None, None

def get_rank_color(rank):
    if rank == 1:
        return "🥇"
    elif rank == 2:
        return "🥈"
    elif rank == 3:
        return "🥉"
    else:
        return f"#{rank}"

def test_all_models(available_models, test_path, device, img_size, batch_size):
    if not available_models:
        st.warning("⚠️ Модели не найдены в папке models/")
        return []

    all_results = []
    all_test_infos = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, model_file in enumerate(available_models):
        try:
            progress = (i + 1) / len(available_models)
            progress_bar.progress(progress)
            status_text.text(f'🔄 Тестирую {i+1}/{len(available_models)}: {os.path.basename(model_file)}')

            test_model = models.resnet18()
            test_model.fc = nn.Linear(test_model.fc.in_features, len(CLASSES))
            state = torch.load(model_file, map_location=device)
            test_model.load_state_dict(state)
            test_model = test_model.to(device)
            test_model.eval()

            metrics, test_info = run_model_tests(
                model=test_model,
                model_path=model_file,
                test_path=test_path,
                device=device,
                img_size=img_size,
                batch_size=batch_size
            )

            all_results.append({
                'model': os.path.basename(model_file),
                'model_path': model_file,
                'accuracy': metrics.accuracy,
                'f1_macro': metrics.f1_macro,
                'roc_auc_avg': np.mean(list(metrics.roc_auc.values())),
                'roc_auc': metrics.roc_auc,
                'confusion': metrics.confusion,
                'report': metrics.report,
                'fig_roc': metrics.fig_roc,
                'error': None,
                'test_info': test_info
            })

            all_test_infos.append(test_info)

        except Exception as e:
            all_results.append({
                'model': os.path.basename(model_file),
                'model_path': model_file,
                'accuracy': 0.0,
                'f1_macro': 0.0,
                'roc_auc_avg': 0.0,
                'roc_auc': {},
                'confusion': None,
                'report': '',
                'fig_roc': None,
                'error': str(e),
                'test_info': None
            })

    progress_bar.progress(1.0)
    status_text.text('✅ Все модели протестированы!')

    return all_results, all_test_infos

def create_comparison_visualization(all_results):
    if not all_results:
        return None, None

    df = pd.DataFrame([
        {
            'Model': result['model'],
            'Accuracy': result['accuracy'],
            'F1 Macro': result['f1_macro'],
            'Avg ROC AUC': result['roc_auc_avg']
        }
        for result in all_results
        if 'error' not in result
    ])

    df = df.sort_values('Accuracy', ascending=False)
    df['Rank'] = range(1, len(df) + 1)

    winner = df.iloc[0]

    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(df)))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    bars1 = ax1.bar(range(len(df)), df['Accuracy'], width=0.6, color=colors)
    ax1.set_title('🏆 Сравнение моделей по Accuracy', fontweight='bold')
    ax1.set_xlabel('Модели')
    ax1.set_ylabel('Accuracy')
    ax1.set_xticks(range(len(df)))
    ax1.set_xticklabels(df['Model'], rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)

    bars2 = ax2.bar(range(len(df)), df['F1 Macro'], width=0.6, color=colors)
    ax2.set_title('🥈 Сравнение моделей по F1 Macro', fontweight='bold')
    ax2.set_xlabel('Модели')
    ax2.set_ylabel('F1 Macro')
    ax2.set_xticks(range(len(df)))
    ax2.set_xticklabels(df['Model'], rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, df, winner

def export_results(df, winner):

    export_df = df.copy()
    export_df.columns = ['Rank', 'Model', 'Accuracy', 'F1_Macro', 'Avg_ROC_AUC']

    csv_ranked = export_df.to_csv(index=False)

    stats_df = df[['Model', 'Accuracy', 'F1_Macro', 'Avg_ROC_AUC']].copy()
    stats_df.columns = ['Model', 'Accuracy', 'F1_Macro', 'Avg_ROC_AUC']

    csv_stats = stats_df.to_csv(index=False)

    return csv_ranked, csv_stats

def process_zip_file(zip_file, model, device, img_size, detect_faces_flag, scale_factor, min_neighbors):

    with tempfile.TemporaryDirectory() as temp_dir:

        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []

        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                if os.path.splitext(file.lower())[1] in image_extensions:
                    image_files.append(os.path.join(root, file))

        if not image_files:
            return None, "В архиве не найдено изображений"

        results = []
        total_faces = 0

        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, img_path in enumerate(image_files):
            try:

                progress = (i + 1) / len(image_files)
                progress_bar.progress(progress)
                status_text.text(f'Обработка {i+1}/{len(image_files)}: {os.path.basename(img_path)}')

                img = Image.open(img_path).convert('RGB')
                img_rgb = np.array(img)
                img_bgr = rgb_to_bgr(img_rgb)

                if detect_faces_flag:
                    faces = detect_faces(img_bgr, scale_factor=scale_factor, min_neighbors=min_neighbors)

                    if len(faces) > 0:
                        for (x, y, w, h) in faces:
                            roi = img_bgr[y:y+h, x:x+w]
                            if roi.size > 0:
                                probs = predict_on_bgr_image(roi, model=model, device=device, img_size=img_size)
                                pred_idx = int(np.argmax(probs))
                                label = CLASSES[pred_idx]
                                conf = float(probs[pred_idx])

                                results.append({
                                    'file': os.path.basename(img_path),
                                    'emotion': label,
                                    'confidence': conf,
                                    'face_count': len(faces)
                                })
                                total_faces += 1
                    else:

                        probs = predict_on_bgr_image(img_bgr, model=model, device=device, img_size=img_size)
                        pred_idx = int(np.argmax(probs))
                        label = CLASSES[pred_idx]
                        conf = float(probs[pred_idx])

                        results.append({
                            'file': os.path.basename(img_path),
                            'emotion': label,
                            'confidence': conf,
                            'face_count': 0
                        })
                else:

                    probs = predict_on_bgr_image(img_bgr, model=model, device=device, img_size=img_size)
                    pred_idx = int(np.argmax(probs))
                    label = CLASSES[pred_idx]
                    conf = float(probs[pred_idx])

                    results.append({
                        'file': os.path.basename(img_path),
                        'emotion': label,
                        'confidence': conf,
                        'face_count': 1
                    })
                    total_faces += 1

            except Exception as e:
                results.append({
                    'file': os.path.basename(img_path),
                    'emotion': 'error',
                    'confidence': 0.0,
                    'face_count': 0,
                    'error': str(e)
                })

        progress_bar.progress(1.0)
        status_text.text('Обработка завершена!')

        return results, None

@dataclass
class TestMetrics:
    accuracy: float
    f1_macro: float
    confusion: np.ndarray
    report: str
    roc_auc: Dict[str, float]
    fig_roc: "object"

def compute_test_metrics(
    model: torch.nn.Module,
    device: torch.device,
    test_path: str,
    img_size: int,
    batch_size: int,
) -> TestMetrics:
    import matplotlib.pyplot as plt

    data_transform = transforms.Compose(
        [
            transforms.Resize(img_size + 32),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    ds = datasets.ImageFolder(test_path, data_transform)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    all_labels: List[int] = []
    all_preds: List[int] = []
    all_probs: List[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_labels.extend(labels.detach().cpu().numpy().tolist())
            all_preds.extend(preds.detach().cpu().numpy().tolist())
            all_probs.extend(probs.detach().cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    acc = float(np.mean(y_true == y_pred))
    f1m = float(f1_score(y_true, y_pred, average='macro'))
    cm = confusion_matrix(y_true, y_pred)
    rep = classification_report(y_true, y_pred, target_names=CLASSES)

    y_true_bin = label_binarize(y_true, classes=list(range(len(CLASSES))))
    roc_auc: Dict[str, float] = {}

    fig = plt.figure()
    for i, cls in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc[cls] = float(auc(fpr, tpr))
        plt.plot(fpr, tpr, label=f'{cls} (AUC = {roc_auc[cls]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')

    return TestMetrics(
        accuracy=acc,
        f1_macro=f1m,
        confusion=cm,
        report=rep,
        roc_auc=roc_auc,
        fig_roc=fig,
    )

def read_saved_metrics(metrics_dir: str) -> Dict[str, object]:
    out: Dict[str, object] = {}

    acc_path = os.path.join(metrics_dir, 'accuracy.txt')
    f1_path = os.path.join(metrics_dir, 'f1_score.txt')
    cm_path = os.path.join(metrics_dir, 'confusion_matrix.txt')
    roc_json = os.path.join(metrics_dir, 'roc_auc.json')
    rep_path = os.path.join(metrics_dir, 'classification_report.txt')
    roc_img = os.path.join(metrics_dir, 'roc_curve.png')

    if os.path.isfile(acc_path):
        with open(acc_path, 'r', encoding='utf-8') as f:
            out['accuracy'] = float(f.read().strip())
    if os.path.isfile(f1_path):
        with open(f1_path, 'r', encoding='utf-8') as f:
            out['f1_score'] = float(f.read().strip())
    if os.path.isfile(cm_path):
        try:
            out['confusion_matrix'] = np.loadtxt(cm_path, dtype=int)
        except Exception:
            pass
    if os.path.isfile(roc_json):
        with open(roc_json, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        out['roc_auc'] = {CLASSES[int(k)]: float(v) for k, v in raw.items()} if raw else {}
    if os.path.isfile(rep_path):
        with open(rep_path, 'r', encoding='utf-8') as f:
            out['classification_report'] = f.read()
    if os.path.isfile(roc_img):
        out['roc_curve_png'] = roc_img

    return out

def main() -> None:
    st.set_page_config(page_title='Анализ эмоций', layout='wide')

    st.title('Распознавание эмоций по лицу')

    available_models = get_available_models()
    available_test_sets = get_available_test_sets()

    with st.sidebar:
        st.header('Настройки')

        if available_models:
            model_options = [os.path.basename(m) for m in available_models]
            selected_model_name = st.selectbox('Выберите модель:', model_options)
            model_path = available_models[model_options.index(selected_model_name)]
        else:
            st.warning("Модели не найдены в папке models/")
            model_path = os.environ.get('MODEL_PATH', 'models/wafflelover404_emotion_model.pth')

        if available_test_sets:
            selected_test_set_name = st.selectbox('Выберите тестовый набор:', list(available_test_sets.keys()))
            test_path = available_test_sets[selected_test_set_name]
        else:
            st.warning("Тестовые наборы не найдены")
            test_path = "test"

        st.write("**Или загрузите свой тестовый набор:**")
        uploaded_test = st.file_uploader('ZIP с тестовыми данными', type=['zip'], key='test_upload')

        if uploaded_test is not None:
            with tempfile.TemporaryDirectory() as temp_dir:
                with zipfile.ZipFile(uploaded_test, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                test_path = temp_dir
                st.success(f"Загружен пользовательский тестовый набор: {temp_dir}")

        img_size = int(os.environ.get('IMG_SIZE', '128'))
        detect_faces_flag = st.toggle('Искать лица', value=True)
        show_probs = st.toggle('Показывать таблицу вероятностей', value=True)
        scale_factor = st.slider('Масштаб (детекция лиц)', min_value=1.05, max_value=1.50, value=1.10, step=0.01)
        min_neighbors = st.slider('Минимальные соседи', min_value=3, max_value=10, value=5, step=1)
        batch_size = st.number_input('Batch size', min_value=1, max_value=128, value=32)

    if not os.path.isfile(model_path):
        st.error('Файл модели не найден. Выберите модель или загрузите модель в папку models/.')
        st.stop()

    model, device = load_model(model_path=model_path, num_classes=len(CLASSES))

    tab_single, tab_batch, tab_metrics, tab_testing, tab_camera = st.tabs(['Одиночное изображение', 'Пакетная обработка ZIP', 'Метрики и ROC', 'Тестирование модели', 'Камера'])

    with tab_single:
        st.subheader('Анализ одного изображения')

        uploaded_file = st.file_uploader('Выберите изображение (jpg/png)', type=['jpg', 'jpeg', 'png'])

        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file).convert('RGB')
                st.success("Изображение загружено")

                img_rgb = np.array(image)
                img_bgr = rgb_to_bgr(img_rgb)

                col_img, col_out = st.columns([2, 1])

                with col_img:
                    st.image(img_rgb, caption='Исходное изображение', use_container_width=True)

                if detect_faces_flag:
                    faces = detect_faces(img_bgr, scale_factor=scale_factor, min_neighbors=min_neighbors)
                    probs_list: List[np.ndarray] = []

                    if len(faces) > 0:
                        for (x, y, w, h) in faces:
                            roi = img_bgr[y : y + h, x : x + w]
                            if roi.size > 0:
                                probs = predict_on_bgr_image(roi, model=model, device=device, img_size=img_size)
                                probs_list.append(probs)

                        if probs_list:
                            overlay = overlay_prediction(img_bgr, faces, probs_list)
                            with col_img:
                                st.image(overlay, caption='Результат детекции', use_container_width=True, channels='RGB')

                            if show_probs:
                                with col_out:
                                    st.subheader("Результаты:")
                                    for i, ((x, y, w, h), probs) in enumerate(zip(faces, probs_list)):
                                        pred_idx = int(np.argmax(probs))
                                        label = CLASSES[pred_idx]
                                        conf = float(probs[pred_idx])

                                        st.write(f"**Лицо {i+1}:** {label}")
                                        st.write(f"**Уверенность:** {conf:.3f}")
                                        st.write("**Вероятности:**")
                                        st.table(probs_to_table(probs))
                                        st.divider()
                        else:
                            with col_out:
                                st.warning("Лица найдены, но не удалось обработать ROI")
                    else:
                        with col_out:
                            st.info("Лица на изображении не найдены")

                            st.write("Анализ всего изображения:")
                            probs = predict_on_bgr_image(img_bgr, model=model, device=device, img_size=img_size)
                            pred_idx = int(np.argmax(probs))
                            label = CLASSES[pred_idx]
                            conf = float(probs[pred_idx])

                            st.write(f"**Эмоция:** {label}")
                            st.write(f"**Уверенность:** {conf:.3f}")
                            if show_probs:
                                st.write("**Вероятности:**")
                                st.table(probs_to_table(probs))
                else:

                    with col_out:
                        st.write("Анализ всего изображения:")
                        probs = predict_on_bgr_image(img_bgr, model=model, device=device, img_size=img_size)
                        pred_idx = int(np.argmax(probs))
                        label = CLASSES[pred_idx]
                        conf = float(probs[pred_idx])

                        st.write(f"**Эмоция:** {label}")
                        st.write(f"**Уверенность:** {conf:.3f}")
                        if show_probs:
                            st.write("**Вероятности:**")
                            st.table(probs_to_table(probs))
            except Exception as e:
                st.error(f"Ошибка при обработке изображения: {e}")
        else:
            st.info("Загрузите изображение для анализа")

    with tab_batch:
        st.subheader('Пакетная обработка ZIP архива')

        zip_file = st.file_uploader('Выберите ZIP архив с изображениями', type=['zip'])

        if zip_file is not None:
            st.info("Начинаю обработку ZIP архива...")

            results, error = process_zip_file(
                zip_file, model, device, img_size,
                detect_faces_flag, scale_factor, min_neighbors
            )

            if error:
                st.error(error)
            elif results:
                st.success(f"Обработано {len(results)} изображений!")

                df_data = []
                for result in results:
                    if result['emotion'] != 'error':
                        df_data.append(result)

                if df_data:
                    df = pd.DataFrame(df_data)

                    st.subheader("Общая статистика")
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("Всего изображений", len(df))
                    with col2:
                        st.metric("Найдено лиц", df['face_count'].sum())
                    with col3:
                        avg_conf = df[df['emotion'] != 'error']['confidence'].mean()
                        st.metric("Средняя уверенность", f"{avg_conf:.3f}")
                    with col4:
                        error_count = len(results) - len(df)
                        st.metric("Ошибок обработки", error_count)

                    st.subheader("Распределение эмоций")
                    emotion_counts = df['emotion'].value_counts()
                    st.bar_chart(emotion_counts)

                    st.subheader("Детальные результаты")
                    st.dataframe(df, use_container_width=True)

                    st.subheader("Распределение уверенности")
                    fig, ax = plt.subplots()
                    ax.hist(df['confidence'], bins=20, alpha=0.7)
                    ax.set_xlabel('Уверенность')
                    ax.set_ylabel('Количество')
                    ax.set_title('Распределение уверенности предсказаний')
                    st.pyplot(fig)

                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Скачать результаты (CSV)",
                        data=csv,
                        file_name="emotion_analysis_results.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("Нет успешно обработанных изображений")

    with tab_metrics:
        st.subheader('Метрики и ROC модели')

        col_test1, col_test2 = st.columns([2, 1])
        with col_test1:
            test_path = st.text_input('Путь к тестовому набору', value='test', key='metrics_test_path')
        with col_test2:
            st.metric("Batch size", batch_size)

        calculate_button = st.button('📊 Вычислить метрики', type='primary')

        if calculate_button:
            if not os.path.isdir(test_path):
                st.error('Папка test не найдена. Укажите корректный путь.')
            else:
                start_time = pd.Timestamp.now()

                with st.spinner('🔄 Вычисляю метрики...'):
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    status_text.text('📂 Подготовка тестовых данных...')
                    progress_bar.progress(0.1)

                    data_transform = transforms.Compose([
                        transforms.Resize(img_size + 32),
                        transforms.CenterCrop(img_size),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    ])

                    ds = datasets.ImageFolder(test_path, data_transform)
                    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

                    total_images = len(ds)
                    expected_batches = (total_images + batch_size - 1) // batch_size
                    status_text.text(f'📊 Найдено изображений: {total_images}')
                    progress_bar.progress(0.2)

                    status_text.text('🧠 Выполнение предсказаний...')
                    
                    all_labels: List[int] = []
                    all_preds: List[int] = []
                    all_probs: List[np.ndarray] = []

                    model.eval()
                    batch_count = 0
                    
                    with torch.no_grad():
                        for inputs, labels in loader:
                            inputs = inputs.to(device)
                            labels = labels.to(device)
                            logits = model(inputs)
                            probs = torch.softmax(logits, dim=1)
                            preds = torch.argmax(probs, dim=1)

                            all_labels.extend(labels.detach().cpu().numpy().tolist())
                            all_preds.extend(preds.detach().cpu().numpy().tolist())
                            all_probs.extend(probs.detach().cpu().numpy())

                            batch_count += 1
                            progress = 0.2 + (batch_count / len(loader)) * 0.5
                            progress_bar.progress(progress)
                            status_text.text(f'🧠 Обработано {batch_count}/{len(loader)} батчей...')

                    status_text.text('📈 Вычисление метрик...')
                    progress_bar.progress(0.8)

                    y_true = np.array(all_labels)
                    y_pred = np.array(all_preds)
                    y_prob = np.array(all_probs)

                    acc = float(np.mean(y_true == y_pred))
                    f1m = float(f1_score(y_true, y_pred, average='macro'))
                    cm = confusion_matrix(y_true, y_pred)
                    rep = classification_report(y_true, y_pred, target_names=CLASSES)

                    y_true_bin = label_binarize(y_true, classes=list(range(len(CLASSES))))
                    roc_auc: Dict[str, float] = {}

                    fig_roc = plt.figure(figsize=(10, 8))
                    for i, cls in enumerate(CLASSES):
                        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
                        roc_auc[cls] = float(auc(fpr, tpr))
                        plt.plot(fpr, tpr, label=f'{cls} (AUC = {roc_auc[cls]:.2f})')
                    plt.plot([0, 1], [0, 1], 'k--')
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('ROC Curve')
                    plt.legend(loc='lower right', fontsize=8)

                    progress_bar.progress(1.0)
                    end_time = pd.Timestamp.now()
                    duration = (end_time - start_time).total_seconds()

                st.success(f"✅ Метрики вычислены за {duration:.2f} сек")

                st.markdown("---")
                st.markdown("### 📊 Основные метрики")
                
                col_a, col_b, col_c = st.columns(3)
                col_a.metric('Accuracy', f'{acc:.4f}')
                col_b.metric('F1 Macro', f'{f1m:.4f}')
                col_c.metric('Avg ROC AUC', f'{np.mean(list(roc_auc.values())):.4f}')

                col_info1, col_info2 = st.columns(2)
                with col_info1:
                    st.metric("Всего изображений", total_images)
                with col_info2:
                    st.metric("Скорость", f"{total_images/duration:.1f} img/сек")

                st.markdown("---")
                st.markdown("### 📈 ROC AUC по классам")
                
                roc_df = pd.DataFrame([
                    {'Класс': cls, 'ROC AUC': auc_val}
                    for cls, auc_val in roc_auc.items()
                ])
                st.dataframe(roc_df, use_container_width=True, hide_index=True)

                st.markdown("### 🎯 ROC Curve")
                st.pyplot(fig_roc)

                st.markdown("### 🔥 Confusion Matrix")
                fig_cm, ax_cm = plt.subplots(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                           xticklabels=CLASSES, yticklabels=CLASSES, ax=ax_cm)
                ax_cm.set_title('Confusion Matrix')
                ax_cm.set_xlabel('Predicted')
                ax_cm.set_ylabel('Actual')
                st.pyplot(fig_cm)

                st.markdown("### 📋 Classification Report")
                st.text(rep)

                st.session_state['last_metrics'] = {
                    'accuracy': acc,
                    'f1_macro': f1m,
                    'roc_auc': roc_auc,
                    'confusion': cm,
                    'report': rep,
                    'fig_roc': fig_roc,
                    'test_path': test_path,
                    'total_images': total_images,
                    'duration': duration
                }

        elif 'last_metrics' in st.session_state:
            st.markdown("---")
            st.markdown("### 📊 Последние вычисленные метрики")
            
            m = st.session_state['last_metrics']
            col_a, col_b, col_c = st.columns(3)
            col_a.metric('Accuracy', f"{m['accuracy']:.4f}")
            col_b.metric('F1 Macro', f"{m['f1_macro']:.4f}")
            col_c.metric('Avg ROC AUC', f"{np.mean(list(m['roc_auc'].values())):.4f}")
            
            st.info(f"Последний тест: {m['test_path']} | Изображений: {m['total_images']} | Время: {m['duration']:.2f} сек")
            
            st.markdown("### 📈 ROC AUC по классам")
            roc_df = pd.DataFrame([
                {'Класс': cls, 'ROC AUC': auc_val}
                for cls, auc_val in m['roc_auc'].items()
            ])
            st.dataframe(roc_df, use_container_width=True, hide_index=True)

            st.markdown("### 🎯 ROC Curve")
            st.pyplot(m['fig_roc'])

            st.markdown("### 🔥 Confusion Matrix")
            fig_cm, ax_cm = plt.subplots(figsize=(10, 8))
            sns.heatmap(m['confusion'], annot=True, fmt='d', cmap='Blues',
                       xticklabels=CLASSES, yticklabels=CLASSES, ax=ax_cm)
            ax_cm.set_title('Confusion Matrix')
            ax_cm.set_xlabel('Predicted')
            ax_cm.set_ylabel('Actual')
            st.pyplot(fig_cm)

            st.markdown("### 📋 Classification Report")
            st.text(m['report'])

        else:
            st.info("👆 Нажмите кнопку 'Вычислить метрики' для расчета метрик модели на тестовом наборе")

    with tab_testing:
        st.subheader('Тестирование модели')

        st.subheader("🎯 Тестирование выбранной модели")

        col1, col2 = st.columns([2, 1])
        with col1:
            st.write("**Текущая модель:**", os.path.basename(model_path))
        with col2:
            if st.button('🚀 Тестировать на всех изображениях', type='primary', key='test_selected_model'):
                if os.path.exists("test") and os.path.isdir("test"):
                    with st.spinner(f'🔄 Тестирую модель {os.path.basename(model_path)} на всех изображениях...'):
                        try:

                            total_images = 0
                            emotion_counts = {}

                            for emotion in CLASSES:
                                emotion_path = os.path.join("test", emotion)
                                if os.path.exists(emotion_path):
                                    count = len([f for f in os.listdir(emotion_path)
                                               if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                                    emotion_counts[emotion] = count
                                    total_images += count

                            if total_images > 0:
                                st.info(f"📊 Найдено изображений: {total_images}")
                                st.write("**Распределение по классам:**")
                                for emotion, count in emotion_counts.items():
                                    if count > 0:
                                        st.write(f"- {emotion}: {count} изображений")

                                metrics, test_info = run_model_tests(
                                    model=model,
                                    model_path=model_path,
                                    test_path="test",
                                    device=device,
                                    img_size=img_size,
                                    batch_size=batch_size
                                )

                                st.success("✅ Тестирование завершено!")

                                roc_auc_avg = np.mean(list(metrics.roc_auc.values())) if metrics.roc_auc else 0.0
                                inference_time = test_info.get('duration_seconds', 0.0)

                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("🎯 Accuracy", f"{metrics.accuracy:.4f}")
                                with col2:
                                    st.metric("📊 F1 Macro", f"{metrics.f1_macro:.4f}")
                                with col3:
                                    st.metric("📈 ROC AUC", f"{roc_auc_avg:.4f}")
                                with col4:
                                    st.metric("⏱️ Время", f"{inference_time:.2f}s")

                                st.subheader("📋 Classification Report")
                                st.text(metrics.report)

                                st.subheader("🔥 Confusion Matrix")
                                fig, ax = plt.subplots(figsize=(8, 6))
                                sns.heatmap(metrics.confusion_matrix,
                                          annot=True, fmt='d', cmap='Blues',
                                          xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
                                ax.set_title('Confusion Matrix')
                                ax.set_xlabel('Predicted')
                                ax.set_ylabel('Actual')
                                st.pyplot(fig)

                                st.subheader("📈 ROC Curves")
                                if hasattr(metrics.fig_roc, 'savefig'):
                                    st.pyplot(metrics.fig_roc)
                                else:
                                    st.info("ROC curves не доступны")

                            else:
                                st.warning("Папка test пуста или не содержит изображений")

                        except Exception as e:
                            st.error(f"❌ Ошибка при тестировании: {e}")
                            import traceback
                            st.error(f"Детали ошибки: {traceback.format_exc()}")
                else:
                    st.error("Стандартная папка test не найдена")

        st.divider()

        if st.button('Проверить стандартный test', key='check_standard_test'):
            if os.path.exists("test") and os.path.isdir("test"):
                st.info("Стандартный тестовый набор найден и готов к использованию")

                total_images = 0
                emotion_counts = {}

                for emotion in CLASSES:
                    emotion_path = os.path.join("test", emotion)
                    if os.path.exists(emotion_path):
                        count = len([f for f in os.listdir(emotion_path)
                                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                        emotion_counts[emotion] = count
                        total_images += count

                if total_images > 0:
                    st.write(f"**Всего изображений в test:** {total_images}")
                    st.write("**Распределение по классам:**")
                    for emotion, count in emotion_counts.items():
                        if count > 0:
                            st.write(f"- {emotion}: {count} изображений")
                else:
                    st.warning("Папка test пуста или не содержит изображений")
            else:
                st.error("Стандартная папка test не найдена")

        st.divider()

        st.subheader("Сравнительное тестирование всех моделей")

        if st.button('🚀 Тестировать все модели', type='primary', key='test_all_models'):
            available_models = get_available_models()

            if available_models:
                all_results, all_test_infos = test_all_models(
                    available_models=available_models,
                    test_path=test_path,
                    device=device,
                    img_size=img_size,
                    batch_size=batch_size
                )

                if all_results:
                    st.subheader("Результаты тестирования всех моделей")

                    df = pd.DataFrame(all_results)

                    df = df.sort_values('accuracy', ascending=False)
                    df['rank'] = range(1, len(df) + 1)

                    winner = df.iloc[0]

                    def get_rank_color(rank):
                        if rank == 1:
                            return "🥇"
                        elif rank == 2:
                            return "🥈"
                        elif rank == 3:
                            return "🥉"
                        else:
                            return f"#{rank}"

                    df['rank_emoji'] = df['rank'].apply(get_rank_color)

                    display_df = df[['rank_emoji', 'model', 'accuracy', 'f1_macro', 'roc_auc_avg']].copy()
                    display_df.columns = ['Ранг', 'Модель', 'Accuracy', 'F1 Macro', 'Avg ROC AUC']

                    st.dataframe(display_df, use_container_width=True)

                    st.success(f"🏆 **Победитель:** {winner['model']} (Accuracy: {winner['accuracy']:.4f}, F1: {winner['f1_macro']:.4f})")

                    st.subheader("Сравнение моделей с ранжированием")
                    fig, ax = plt.subplots(figsize=(12, 7))

                    models_names = df['model'].tolist()
                    accuracies = df['accuracy'].tolist()
                    f1_scores = df['f1_macro'].tolist()
                    ranks = df['rank'].tolist()

                    x = np.arange(len(models_names))
                    width = 0.6

                    colors_list = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(models_names)))

                    bars = ax.bar(x, accuracies, width, label='Accuracy', color=colors_list)

                    for i, (rank, acc, model_name) in enumerate(zip(ranks, accuracies, models_names)):
                        ax.text(i + width/2, acc + 0.01, f'#{rank}',
                                ha='center', va='bottom', fontweight='bold', fontsize=8)
                        ax.text(i + width/2, acc - 0.02, f'{acc:.3f}',
                                ha='center', va='top', fontsize=7)

                    ax.set_xlabel('Модели (ранжированы по Accuracy)')
                    ax.set_ylabel('Accuracy')
                    ax.set_title('Сравнение моделей (🥇 = лучшая)')
                    ax.set_xticks(x)
                    ax.set_xticklabels(models_names, rotation=45, ha='right')
                    ax.legend()
                    ax.grid(True, alpha=0.3)

                    rank_colors = plt.cm.RdYlGn(np.linspace(0.8, 0.2, len(ranks)))
                    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn)
                    sm.set_array(np.array(ranks))
                    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.02, pad=0.04)
                    cbar.set_label('Ранг (1=лучший)', rotation=270, va='bottom')
                    cbar.set_ticks(ranks)

                    st.pyplot(fig)

                    st.subheader("📋 Детальная статистика по моделям")

                    for idx, row in df.iterrows():
                        with st.expander(f"🏆 #{row['rank']} {row['model']} - Accuracy: {row['accuracy']:.4f}"):
                            col1, col2, col3 = st.columns(3)

                            with col1:
                                st.metric("Accuracy", f"{row['accuracy']:.4f}")
                                st.metric("F1 Macro", f"{row['f1_macro']:.4f}")

                            with col2:
                                st.metric("Avg ROC AUC", f"{row['roc_auc_avg']:.4f}")
                                st.metric("Ранг", f"#{row['rank']}")

                            with col3:
                                if row['rank'] == 1:
                                    st.success("🥇 Лучшая модель")
                                elif row['rank'] == 2:
                                    st.info("🥈 Второе место")
                                elif row['rank'] == 3:
                                    st.warning("🥉 Третье место")
                                else:
                                    st.write(f"Место: #{row['rank']}")

                            if 'roc_auc' in row and isinstance(row['roc_auc'], dict):
                                st.write("**ROC AUC по классам:**")
                                roc_auc_df = pd.DataFrame([
                                    {'Класс': k, 'ROC AUC': f"{v:.4f}"}
                                    for k, v in row['roc_auc'].items()
                                ])
                                st.dataframe(roc_auc_df, use_container_width=True)

                    st.divider()

                    st.subheader("💾 Экспорт результатов")
                    col_export1, col_export2, col_export3, col_export4 = st.columns(4)

                    with col_export1:

                        detailed_csv_data = []

                        headers = [
                            'Rank', 'Model', 'Accuracy', 'F1_Macro', 'Avg_ROC_AUC',
                            'ROC_AUC_angry', 'ROC_AUC_disgust', 'ROC_AUC_fear', 'ROC_AUC_happy',
                            'ROC_AUC_neutral', 'ROC_AUC_sad', 'ROC_AUC_surprise',
                            'Precision_angry', 'Recall_angry', 'F1_angry',
                            'Precision_disgust', 'Recall_disgust', 'F1_disgust',
                            'Precision_fear', 'Recall_fear', 'F1_fear',
                            'Precision_happy', 'Recall_happy', 'F1_happy',
                            'Precision_neutral', 'Recall_neutral', 'F1_neutral',
                            'Precision_sad', 'Recall_sad', 'F1_sad',
                            'Precision_surprise', 'Recall_surprise', 'F1_surprise',
                            'True_angry', 'True_disgust', 'True_fear', 'True_happy',
                            'True_neutral', 'True_sad', 'True_surprise',
                            'Pred_angry', 'Pred_disgust', 'Pred_fear', 'Pred_happy',
                            'Pred_neutral', 'Pred_sad', 'Pred_surprise',
                            'Support_angry', 'Support_disgust', 'Support_fear', 'Support_happy',
                            'Support_neutral', 'Support_sad', 'Support_surprise',
                            'Error_Status'
                        ]
                        detailed_csv_data.append(','.join(headers))

                        for idx, row in df.iterrows():
                            model_result = all_results[idx]
                            model_name = row['model']

                            base_data = [
                                str(row['rank']),
                                f'"{model_name}"',
                                f"{row['accuracy']:.6f}",
                                f"{row['f1_macro']:.6f}",
                                f"{row['roc_auc_avg']:.6f}"
                            ]

                            roc_auc_data = []
                            if 'roc_auc' in model_result and isinstance(model_result['roc_auc'], dict):
                                for cls in CLASSES:
                                    roc_auc_data.append(f"{model_result['roc_auc'].get(cls, 0):.6f}")
                            else:
                                roc_auc_data = ['0.000000'] * len(CLASSES)

                            precision_data = []
                            recall_data = []
                            f1_data = []
                            support_data = []

                            if 'report' in model_result and model_result['report']:
                                report_lines = model_result['report'].split('\n')
                                for line in report_lines:
                                    if any(cls in line for cls in CLASSES):
                                        parts = line.split()
                                        if len(parts) >= 4:
                                            try:

                                                cls_name = None
                                                for cls in CLASSES:
                                                    if line.strip().startswith(cls):
                                                        cls_name = cls
                                                        break

                                                if cls_name:

                                                    precision = float(parts[-4])
                                                    recall = float(parts[-3])
                                                    f1 = float(parts[-2])
                                                    support = int(parts[-1])

                                                    precision_data.append(f"{precision:.6f}")
                                                    recall_data.append(f"{recall:.6f}")
                                                    f1_data.append(f"{f1:.6f}")
                                                    support_data.append(str(support))
                                            except (ValueError, IndexError):

                                                precision_data.append("0.000000")
                                                recall_data.append("0.000000")
                                                f1_data.append("0.000000")
                                                support_data.append("0")

                            while len(precision_data) < len(CLASSES):
                                precision_data.append("0.000000")
                                recall_data.append("0.000000")
                                f1_data.append("0.000000")
                                support_data.append("0")

                            confusion_data = []
                            if 'confusion' in model_result and model_result['confusion'] is not None:
                                cm = model_result['confusion']
                                if isinstance(cm, np.ndarray):

                                    for i in range(len(CLASSES)):
                                        confusion_data.append(str(int(cm[i].sum())))

                                    for j in range(len(CLASSES)):
                                        confusion_data.append(str(int(cm[:, j].sum())))
                            else:
                                confusion_data = ['0'] * (len(CLASSES) * 2)

                            error_status = model_result.get('error', 'Success')
                            if error_status:
                                error_status = f'"{error_status}"'
                            else:
                                error_status = 'Success'

                            full_row = base_data + roc_auc_data + precision_data + recall_data + f1_data + confusion_data + support_data + [error_status]
                            detailed_csv_data.append(','.join(full_row))

                        detailed_csv_text = '\n'.join(detailed_csv_data)

                        st.download_button(
                            label="📊 Супер детальный CSV",
                            data=detailed_csv_text,
                            file_name="models_super_detailed_report.csv",
                            mime="text/csv"
                        )

                    with col_export2:

                        confusion_csv_data = []

                        confusion_headers = ['Model', 'Rank', 'True_Class', 'Pred_Class', 'Count']
                        confusion_csv_data.append(','.join(confusion_headers))

                        for idx, row in df.iterrows():
                            model_result = all_results[idx]
                            model_name = f'"{row["model"]}"'
                            rank = str(row['rank'])

                            if 'confusion' in model_result and model_result['confusion'] is not None:
                                cm = model_result['confusion']
                                if isinstance(cm, np.ndarray):
                                    for i, true_class in enumerate(CLASSES):
                                        for j, pred_class in enumerate(CLASSES):
                                            count = int(cm[i][j])
                                            if count > 0:
                                                confusion_row = [
                                                    model_name,
                                                    rank,
                                                    true_class,
                                                    pred_class,
                                                    str(count)
                                                ]
                                                confusion_csv_data.append(','.join(confusion_row))

                        confusion_csv_text = '\n'.join(confusion_csv_data)

                        st.download_button(
                            label="🎯 Confusion Matrix CSV",
                            data=confusion_csv_text,
                            file_name="models_confusion_matrices.csv",
                            mime="text/csv"
                        )

                    with col_export3:

                        if st.button("📈 ROC кривые (ZIP)"):
                            import zipfile
                            import io

                            zip_buffer = io.BytesIO()

                            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                                for idx, row in df.iterrows():
                                    model_result = all_results[idx]
                                    model_name = row['model'].replace('.pth', '')

                                    if 'fig_roc' in model_result and model_result['fig_roc']:

                                        img_buffer = io.BytesIO()
                                        model_result['fig_roc'].savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
                                        img_buffer.seek(0)

                                        zip_file.writestr(f"roc_curve_{model_name}.png", img_buffer.getvalue())

                                        metadata = f
                                        if 'roc_auc' in model_result and isinstance(model_result['roc_auc'], dict):
                                            for cls, auc_val in model_result['roc_auc'].items():
                                                metadata += f"{cls}: {auc_val:.6f}\n"

                                        zip_file.writestr(f"roc_metadata_{model_name}.txt", metadata)

                            zip_buffer.seek(0)

                            st.download_button(
                                label="📈 Скачать ROC кривые (ZIP)",
                                data=zip_buffer.getvalue(),
                                file_name="roc_curves.zip",
                                mime="application/zip"
                            )

                    with col_export4:

                        combined_csv_data = []

                        combined_csv_data.append("=== СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ ===")
                        combined_csv_data.append("Rank,Model,Accuracy,F1_Macro,Avg_ROC_AUC")
                        for idx, row in df.iterrows():
                            combined_csv_data.append(f"{row['rank']},\"{row['model']}\",{row['accuracy']:.6f},{row['f1_macro']:.6f},{row['roc_auc_avg']:.6f}")

                        combined_csv_data.append("\n=== ДЕТАЛЬНЫЕ МЕТРИКИ ПО КЛАССАМ ===")

                        class_headers = ['Model', 'Class', 'Precision', 'Recall', 'F1_Score', 'Support', 'ROC_AUC']
                        combined_csv_data.append(','.join(class_headers))

                        for idx, row in df.iterrows():
                            model_result = all_results[idx]
                            model_name = f'"{row["model"]}"'

                            if 'report' in model_result and model_result['report']:
                                report_lines = model_result['report'].split('\n')
                                for line in report_lines:
                                    if any(cls in line for cls in CLASSES):
                                        parts = line.split()
                                        if len(parts) >= 4:
                                            try:

                                                cls_name = None
                                                for cls in CLASSES:
                                                    if line.strip().startswith(cls):
                                                        cls_name = cls
                                                        break

                                                if cls_name:
                                                    precision = float(parts[-4])
                                                    recall = float(parts[-3])
                                                    f1 = float(parts[-2])
                                                    support = int(parts[-1])
                                                    roc_auc = model_result.get('roc_auc', {}).get(cls_name, 0)

                                                    class_row = [
                                                        model_name,
                                                        cls_name,
                                                        f"{precision:.6f}",
                                                        f"{recall:.6f}",
                                                        f"{f1:.6f}",
                                                        str(support),
                                                        f"{roc_auc:.6f}"
                                                    ]
                                                    combined_csv_data.append(','.join(class_row))
                                            except (ValueError, IndexError):
                                                pass

                        combined_csv_data.append("\n=== СТАТИСТИКА ПО ВСЕМ МОДЕЛЯМ ===")
                        accuracies = df['accuracy'].tolist()
                        f1_scores = df['f1_macro'].tolist()
                        roc_aucs = df['roc_auc_avg'].tolist()

                        stats_data = [
                            f"Metric,Mean,Std,Min,Max",
                            f"Accuracy,{np.mean(accuracies):.6f},{np.std(accuracies):.6f},{np.min(accuracies):.6f},{np.max(accuracies):.6f}",
                            f"F1_Macro,{np.mean(f1_scores):.6f},{np.std(f1_scores):.6f},{np.min(f1_scores):.6f},{np.max(f1_scores):.6f}",
                            f"ROC_AUC,{np.mean(roc_aucs):.6f},{np.std(roc_aucs):.6f},{np.min(roc_aucs):.6f},{np.max(roc_aucs):.6f}"
                        ]
                        combined_csv_data.extend(stats_data)

                        combined_csv_text = '\n'.join(combined_csv_data)

                        st.download_button(
                            label="📋 Комбинированный отчет (CSV)",
                            data=combined_csv_text,
                            file_name="models_combined_report.csv",
                            mime="text/csv"
                        )

                    with col_export3:

                        detailed_report = []
                        detailed_report.append("=== СРАВНИТЕЛЬНЫЙ ОТЧЕТ МОДЕЛЕЙ ===\n")
                        detailed_report.append(f"Тестовый набор: {test_path}")
                        detailed_report.append(f"Размер изображения: {img_size}x{img_size}")
                        detailed_report.append(f"Batch size: {batch_size}")
                        detailed_report.append(f"Всего моделей: {len(df)}")
                        detailed_report.append(f"Дата тестирования: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

                        detailed_report.append("=== РАНЖИРОВАНИЕ ===")
                        for idx, row in df.iterrows():
                            detailed_report.append(f"#{row['rank']} {row['rank_emoji']} {row['model']}")
                            detailed_report.append(f"  Accuracy: {row['accuracy']:.4f}")
                            detailed_report.append(f"  F1 Macro: {row['f1_macro']:.4f}")
                            detailed_report.append(f"  Avg ROC AUC: {row['roc_auc_avg']:.4f}\n")

                        detailed_report.append("=== ПОБЕДИТЕЛЬ ===")
                        winner = df.iloc[0]
                        detailed_report.append(f"🏆 Лучшая модель: {winner['model']}")
                        detailed_report.append(f"   Accuracy: {winner['accuracy']:.4f}")
                        detailed_report.append(f"   F1 Macro: {winner['f1_macro']:.4f}")
                        detailed_report.append(f"   Avg ROC AUC: {winner['roc_auc_avg']:.4f}")

                        report_text = "\n".join(detailed_report)

                        st.download_button(
                            label="📄 Детальный отчет (TXT)",
                            data=report_text,
                            file_name="models_comparison_report.txt",
                            mime="text/plain"
                        )

                    with col_export1:

                        super_detailed_report = []
                        super_detailed_report.append("=" * 80)
                        super_detailed_report.append("СУПЕР ДЕТАЛЬНЫЙ ОТЧЕТ ПО ТЕСТИРОВАНИЮ МОДЕЛЕЙ")
                        super_detailed_report.append("=" * 80)
                        super_detailed_report.append("")

                        super_detailed_report.append("ОБЩАЯ ИНФОРМАЦИЯ")
                        super_detailed_report.append("-" * 40)
                        super_detailed_report.append(f"Тестовый набор: {test_path}")
                        super_detailed_report.append(f"Размер изображения: {img_size}x{img_size}")
                        super_detailed_report.append(f"Batch size: {batch_size}")
                        super_detailed_report.append(f"Всего моделей протестировано: {len(df)}")
                        super_detailed_report.append(f"Дата и время тестирования: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
                        super_detailed_report.append(f"Устройство: {device}")
                        super_detailed_report.append("")

                        super_detailed_report.append("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
                        super_detailed_report.append("-" * 40)
                        super_detailed_report.append(f"{'Ранг':<5} {'Модель':<30} {'Accuracy':<12} {'F1 Macro':<12} {'ROC AUC':<12}")
                        super_detailed_report.append("-" * 80)
                        for idx, row in df.iterrows():
                            super_detailed_report.append(
                                f"#{row['rank']:<4} {row['model'][:29]:<30} {row['accuracy']:<12.4f} "
                                f"{row['f1_macro']:<12.4f} {row['roc_auc_avg']:<12.4f}"
                            )
                        super_detailed_report.append("")

                        super_detailed_report.append("ДЕТАЛЬНАЯ ИНФОРМАЦИЯ ПО КАЖДОЙ МОДЕЛИ")
                        super_detailed_report.append("=" * 80)

                        for idx, row in df.iterrows():
                            model_result = all_results[idx]
                            super_detailed_report.append("")
                            super_detailed_report.append(f"МОДЕЛЬ #{row['rank']}: {row['model']}")
                            super_detailed_report.append("=" * 50)

                            super_detailed_report.append("ОСНОВНЫЕ МЕТРИКИ:")
                            super_detailed_report.append(f"  Accuracy: {row['accuracy']:.6f}")
                            super_detailed_report.append(f"  F1 Macro: {row['f1_macro']:.6f}")
                            super_detailed_report.append(f"  Average ROC AUC: {row['roc_auc_avg']:.6f}")

                            if 'roc_auc' in model_result and isinstance(model_result['roc_auc'], dict):
                                super_detailed_report.append("")
                                super_detailed_report.append("ROC AUC ПО КЛАССАМ:")
                                for class_name, roc_auc_value in model_result['roc_auc'].items():
                                    super_detailed_report.append(f"  {class_name:<12}: {roc_auc_value:.6f}")

                            if 'report' in model_result and model_result['report']:
                                super_detailed_report.append("")
                                super_detailed_report.append("CLASSIFICATION REPORT:")
                                super_detailed_report.append("-" * 30)
                                report_lines = model_result['report'].split('\n')
                                for line in report_lines:
                                    if line.strip():
                                        super_detailed_report.append(f"  {line}")

                            if 'confusion' in model_result and model_result['confusion'] is not None:
                                super_detailed_report.append("")
                                super_detailed_report.append("CONFUSION MATRIX:")
                                super_detailed_report.append("-" * 20)
                                cm = model_result['confusion']
                                if isinstance(cm, np.ndarray):

                                    header = "        " + "  ".join([f"{cls:>8}" for cls in CLASSES])
                                    super_detailed_report.append(header)
                                    super_detailed_report.append("        " + "-" * (8 * len(CLASSES) + 2 * (len(CLASSES) - 1)))

                                    for i, true_class in enumerate(CLASSES):
                                        row_str = f"{true_class:>8}: " + "  ".join([f"{cm[i][j]:>8}" for j in range(len(CLASSES))])
                                        super_detailed_report.append(row_str)

                            super_detailed_report.append("")
                            super_detailed_report.append("ДОПОЛНИТЕЛЬНАЯ СТАТИСТИКА:")
                            if 'error' in model_result and model_result['error']:
                                super_detailed_report.append(f"  Ошибки при тестировании: {model_result['error']}")
                            else:
                                super_detailed_report.append("  Тестирование пройдено успешно")

                            if row['rank'] == 1:
                                super_detailed_report.append("  🥇 ЛУЧШАЯ МОДЕЛЬ В РЕЙТИНГЕ")
                            elif row['rank'] == 2:
                                super_detailed_report.append("  🥈 ВТОРОЕ МЕСТО")
                            elif row['rank'] == 3:
                                super_detailed_report.append("  🥉 ТРЕТЬЕ МЕСТО")
                            else:
                                super_detailed_report.append(f"  Место в рейтинге: #{row['rank']}")

                            super_detailed_report.append("")
                            super_detailed_report.append("~" * 50)

                        super_detailed_report.append("")
                        super_detailed_report.append("ИТОГОВЫЕ ВЫВОДЫ")
                        super_detailed_report.append("=" * 50)
                        super_detailed_report.append(f"🏆 Победитель тестирования: {winner['model']}")
                        super_detailed_report.append(f"   Лучший Accuracy: {winner['accuracy']:.6f}")
                        super_detailed_report.append(f"   Лучший F1 Macro: {winner['f1_macro']:.6f}")
                        super_detailed_report.append(f"   Лучший Avg ROC AUC: {winner['roc_auc_avg']:.6f}")

                        accuracies = df['accuracy'].tolist()
                        f1_scores = df['f1_macro'].tolist()
                        roc_aucs = df['roc_auc_avg'].tolist()

                        super_detailed_report.append("")
                        super_detailed_report.append("СТАТИСТИКА ПО ВСЕМ МОДЕЛЯМ:")
                        super_detailed_report.append(f"  Средний Accuracy: {np.mean(accuracies):.6f} ± {np.std(accuracies):.6f}")
                        super_detailed_report.append(f"  Средний F1 Macro: {np.mean(f1_scores):.6f} ± {np.std(f1_scores):.6f}")
                        super_detailed_report.append(f"  Средний ROC AUC: {np.mean(roc_aucs):.6f} ± {np.std(roc_aucs):.6f}")
                        super_detailed_report.append(f"  Разброс Accuracy: {np.max(accuracies) - np.min(accuracies):.6f}")
                        super_detailed_report.append(f"  Разброс F1 Macro: {np.max(f1_scores) - np.min(f1_scores):.6f}")

                        super_detailed_report.append("")
                        super_detailed_report.append("=" * 80)
                        super_detailed_report.append("КОНЕЦ ОТЧЕТА")
                        super_detailed_report.append("=" * 80)

                        super_report_text = "\n".join(super_detailed_report)

                        st.download_button(
                            label="📋 Супер детальный отчет (TXT)",
                            data=super_report_text,
                            file_name="models_super_detailed_report.txt",
                            mime="text/plain"
                        )

                    with col_export2:

                        json_export = {
                            "test_info": {
                                "test_path": test_path,
                                "img_size": img_size,
                                "batch_size": batch_size,
                                "device": str(device),
                                "timestamp": pd.Timestamp.now().isoformat(),
                                "total_models": len(df)
                            },
                            "ranking": {
                                "winner": {
                                    "model": winner['model'],
                                    "rank": 1,
                                    "accuracy": winner['accuracy'],
                                    "f1_macro": winner['f1_macro'],
                                    "roc_auc_avg": winner['roc_auc_avg']
                                },
                                "all_models": df.to_dict('records')
                            },
                            "detailed_results": [],
                            "test_infos": [],
                            "statistics": {
                                "accuracy": {
                                    "mean": float(np.mean(accuracies)),
                                    "std": float(np.std(accuracies)),
                                    "min": float(np.min(accuracies)),
                                    "max": float(np.max(accuracies))
                                },
                                "f1_macro": {
                                    "mean": float(np.mean(f1_scores)),
                                    "std": float(np.std(f1_scores)),
                                    "min": float(np.min(f1_scores)),
                                    "max": float(np.max(f1_scores))
                                },
                                "roc_auc_avg": {
                                    "mean": float(np.mean(roc_aucs)),
                                    "std": float(np.std(roc_aucs)),
                                    "min": float(np.min(roc_aucs)),
                                    "max": float(np.max(roc_aucs))
                                }
                            }
                        }

                        for result in all_results:
                            result_copy = result.copy()

                            if 'fig_roc' in result_copy:
                                del result_copy['fig_roc']
                            if 'test_info' in result_copy and result_copy['test_info']:
                                test_info_copy = result_copy['test_info'].copy()

                                if 'start_time' in test_info_copy:
                                    test_info_copy['start_time'] = str(test_info_copy['start_time'])
                                if 'end_time' in test_info_copy:
                                    test_info_copy['end_time'] = str(test_info_copy['end_time'])
                                result_copy['test_info'] = test_info_copy
                            json_export["detailed_results"].append(result_copy)

                        for test_info in all_test_infos:
                            if test_info:
                                test_info_copy = test_info.copy()

                                if 'start_time' in test_info_copy:
                                    test_info_copy['start_time'] = str(test_info_copy['start_time'])
                                if 'end_time' in test_info_copy:
                                    test_info_copy['end_time'] = str(test_info_copy['end_time'])
                                json_export["test_infos"].append(test_info_copy)

                        def convert_numpy(obj):
                            try:
                                if isinstance(obj, np.ndarray):
                                    return obj.tolist()
                                elif isinstance(obj, np.integer):
                                    return int(obj)
                                elif isinstance(obj, np.floating):
                                    return float(obj)
                                elif isinstance(obj, dict):
                                    return {key: convert_numpy(value) for key, value in obj.items()}
                                elif isinstance(obj, list):
                                    return [convert_numpy(item) for item in obj]
                                elif hasattr(obj, '__class__'):

                                    class_name = obj.__class__.__name__
                                    if 'Figure' in class_name:
                                        return f"[Figure object - {class_name}]"
                                    elif 'Timestamp' in class_name:
                                        return str(obj)
                                    elif 'Axes' in class_name:
                                        return f"[Axes object - {class_name}]"
                                    elif 'Canvas' in class_name:
                                        return f"[Canvas object - {class_name}]"
                                    else:

                                        return str(obj)
                                else:
                                    return obj
                            except Exception as e:

                                return f"[Conversion error: {str(obj)}]"

                        json_export_clean = convert_numpy(json_export)
                        json_data = json.dumps(json_export_clean, indent=2, ensure_ascii=False)

                        st.download_button(
                            label="📄 Полный отчет (JSON)",
                            data=json_data,
                            file_name="models_complete_report.json",
                            mime="application/json"
                        )

                    st.session_state.last_test_results = all_results
                    st.session_state.last_test_df = df
                    st.session_state.last_test_winner = winner
                    st.session_state.last_test_infos = all_test_infos

                    st.session_state.last_test_path = test_path
                    st.session_state.last_img_size = img_size
                    st.session_state.last_batch_size = batch_size
                    st.session_state.last_device = device

                    try:
                        with st.spinner("🔄 Автоматическое создание ZIP-отчета..."):
                            zip_data = create_complete_zip_report(
                                all_results, df, winner, all_test_infos,
                                test_path, img_size, batch_size, device
                            )

                            os.makedirs('logs', exist_ok=True)
                            zip_filename = f"logs/models_complete_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.zip"

                            with open(zip_filename, 'wb') as f:
                                f.write(zip_data)

                            st.success(f"📦 ZIP-отчет автоматически сохранен: {zip_filename}")
                            st.info(f"📁 Размер файла: {len(zip_data)} байт")
                            st.info(f"📂 Файл доступен в директории: {os.path.abspath(zip_filename)}")
                    except Exception as e:
                        st.error(f"❌ Ошибка при автоматическом создании ZIP-отчета: {e}")
                        import traceback
                        st.error(f"Детали ошибки: {traceback.format_exc()}")

                st.divider()
                st.subheader("🗂️ Универсальный экспорт")
                st.markdown("Скачайте полный ZIP-архив со всеми отчетами и данными")

                if 'last_test_results' in st.session_state and st.session_state.last_test_results:
                    st.success("✅ Результаты тестирования доступны для скачивания")

                    if st.button("📦 Создать и сохранить ZIP-отчет", type="primary", use_container_width=True):
                        try:
                            with st.spinner("🔄 Создание ZIP-архива..."):

                                all_results = st.session_state.last_test_results
                                df = st.session_state.last_test_df
                                winner = st.session_state.last_test_winner
                                all_test_infos = st.session_state.last_test_infos
                                test_path = st.session_state.last_test_path
                                img_size = st.session_state.last_img_size
                                batch_size = st.session_state.last_batch_size
                                device = st.session_state.last_device

                                zip_data = create_complete_zip_report(
                                    all_results, df, winner, all_test_infos,
                                    test_path, img_size, batch_size, device
                                )

                                os.makedirs('logs', exist_ok=True)
                                zip_filename = f"logs/models_complete_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.zip"

                                with open(zip_filename, 'wb') as f:
                                    f.write(zip_data)

                                st.success(f"📦 ZIP-отчет сохранен: {zip_filename}")
                                st.info(f"📁 Размер файла: {len(zip_data)} байт")
                                st.info(f"📂 Файл доступен в директории: {os.path.abspath(zip_filename)}")

                        except Exception as e:
                            st.error(f"❌ Ошибка при создании ZIP-отчета: {e}")
                            import traceback
                            st.error(f"Детали ошибки: {traceback.format_exc()}")
                else:
                    st.info("📋 Для создания отчета сначала проведите тестирование всех моделей")

    with tab_camera:
        st.subheader('📷 Распознавание эмоций с камеры')

        st.write()

        source_option = st.radio(
            "Выберите источник изображения:",
            ["📸 Сделать снимок", "🎥 Живая камера"],
            horizontal=True
        )

        col1, col2 = st.columns(2)
        with col1:
            camera_scale_factor = st.slider('Масштаб (детекция лиц)', min_value=1.05, max_value=1.50, value=1.10, step=0.01)
        with col2:
            camera_min_neighbors = st.slider('Минимальные соседи', min_value=3, max_value=10, value=5, step=1)

        if source_option == "🎥 Живая камера":
            col3, col4 = st.columns(2)
            with col3:
                fps_limit = st.slider('FPS лимит', min_value=1, max_value=30, value=10, step=1)
            with col4:
                show_confidence = st.toggle('Показывать уверенность', value=True)

        face_cascade = get_face_cascade()

        if source_option == "🎥 Живая камера":

            st.info("🎥 Нажмите 'Запустить живую камеру' для начала распознавания в реальном времени")

            if st.button("🎥 Запустить живую камера", type="primary"):

                video_placeholder = st.empty()
                stats_placeholder = st.empty()

                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    st.error("❌ Не удалось открыть камеру. Убедитесь, что камера доступна.")
                    st.stop()

                st.info("🔴 Камера запущена. Нажмите 'Остановить' для завершения.")

                stop_button = st.button("🛑 Остановить камеру")

                frame_count = 0
                start_time = time.time()

                while not stop_button and cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_count += 1
                    if frame_count % (30 // fps_limit) != 0:
                        continue

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    faces = face_cascade.detectMultiScale(
                        gray,
                        scaleFactor=camera_scale_factor,
                        minNeighbors=camera_min_neighbors,
                        minSize=(30, 30)
                    )

                    results = []
                    for (x, y, w, h) in faces:
                        face_roi = frame[y:y+h, x:x+w]

                        probs = predict_on_bgr_image(face_roi, model, device, img_size)
                        emotion_idx = np.argmax(probs)
                        emotion = CLASSES[emotion_idx]
                        confidence = float(probs[emotion_idx])

                        results.append({
                            'emotion': emotion,
                            'confidence': confidence,
                            'position': (x, y, w, h)
                        })

                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                        if show_confidence:
                            label = f'{emotion}: {confidence:.0%}'
                        else:
                            label = emotion

                        cv2.putText(frame, label, (x, y-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                                         channels="RGB", use_container_width=True)

                    if results:
                        emotion_counts = {}
                        for result in results:
                            emotion = result['emotion']
                            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

                        stats_text = f"🎯 Найдено лиц: {len(results)} | "
                        stats_text += " | ".join([f"{emo}: {count}" for emo, count in emotion_counts.items()])

                        stats_placeholder.info(stats_text)
                    else:
                        stats_placeholder.warning("😔 Лица не обнаружены")

                    time.sleep(0.01)

                cap.release()
                video_placeholder.empty()
                stats_placeholder.empty()
                st.success("✅ Камера остановлена")

        elif source_option == "📸 Сделать снимок":

            image_file = st.camera_input("Сделайте снимок для анализа эмоций")

            if image_file is not None:
                process_single_image(image_file, face_cascade, camera_scale_factor,
                                   camera_min_neighbors, model, device, img_size)

        with st.expander("📖 Инструкция по использованию"):
            st.write("""
            **Как использовать камеру для распознавания эмоций:**

            **📸 Сделать снимок:**
            - Нажмите кнопку камеры для захвата одного изображения
            - Автоматический анализ эмоций на сделанном фото

            **🎥 Живая камера:**
            - Нажмите "Запустить живую камеру" для начала стрима
            - Распознавание эмоций в реальном времени
            - Нажмите "Остановить камеру" для завершения

            **Настройки детекции:**
            - Используйте ползунки для настройки параметров детекции лиц
            - Для живой камеры: настройте FPS и отображение уверенности

            **Советы для лучших результатов:**
            - 📸 Убедитесь, что лицо хорошо освещено
            - 👤 Смотрите прямо в камеру
            - 🎯 Избегайте частичного перекрытия лица
            - 📏 Расстояние до камеры: 0.5-2 метра
            - 😊 Выражайте эмоции естественно

            **Поддерживаемые эмоции:**
            - Angry (гнев)
            - Disgust (отвращение)
            - Fear (страх)
            - Happy (счастье)
            - Neutral (нейтральный)
            - Sad (грусть)
            - Surprise (удивление)
            """)

def process_single_image(image_file, face_cascade, scale_factor, min_neighbors, model, device, img_size):
    try:

        image = Image.open(image_file).convert('RGB')
        image_array = np.array(image)
        image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=(30, 30)
        )

        result_image = image_bgr.copy()

        if len(faces) > 0:
            st.success(f"🎯 Найдено лиц: {len(faces)}")

            results = []
            for i, (x, y, w, h) in enumerate(faces):

                face_roi = image_bgr[y:y+h, x:x+w]

                probs = predict_on_bgr_image(face_roi, model, device, img_size)
                emotion_idx = np.argmax(probs)
                emotion = CLASSES[emotion_idx]
                confidence = float(probs[emotion_idx])

                results.append({
                    'face_num': i + 1,
                    'emotion': emotion,
                    'confidence': confidence,
                    'position': (x, y, w, h),
                    'all_probs': {CLASSES[j]: float(probs[j]) for j in range(len(CLASSES))}
                })

                cv2.rectangle(result_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                label = f'{emotion}: {confidence:.2%}'
                cv2.putText(result_image, label, (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            st.image(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB),
                    caption="Результаты детекции", use_container_width=True)

            st.subheader("📊 Детальные результаты")

            for result in results:
                with st.expander(f"Лицо #{result['face_num']} - {result['emotion']} ({result['confidence']:.2%})"):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric("Эмоция", result['emotion'])
                        st.metric("Уверенность", f"{result['confidence']:.2%}")
                        st.write(f"Позиция: x={result['position'][0]}, y={result['position'][1]}")
                        st.write(f"Размер: {result['position'][2]}x{result['position'][3]}")

                    with col2:
                        st.write("**Все вероятности:**")
                        for emotion, prob in result['all_probs'].items():
                            st.write(f"- {emotion}: {prob:.2%}")

            st.subheader("📈 Общая статистика")
            emotion_counts = {}
            for result in results:
                emotion = result['emotion']
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

            if emotion_counts:

                stats_data = [
                    {'Эмоция': emotion, 'Количество': count, 'Процент': f"{count/len(results)*100:.1f}%"}
                    for emotion, count in emotion_counts.items()
                ]
                stats_df = pd.DataFrame(stats_data)
                st.dataframe(stats_df, use_container_width=True)

                fig, ax = plt.subplots(figsize=(8, 4))
                emotions = list(emotion_counts.keys())
                counts = list(emotion_counts.values())

                colors = plt.cm.Set3(np.linspace(0, 1, len(emotions)))
                bars = ax.bar(emotions, counts, color=colors)

                ax.set_xlabel('Эмоции')
                ax.set_ylabel('Количество лиц')
                ax.set_title('Распределение эмоций на фото')
                ax.grid(True, alpha=0.3)

                for bar, count in zip(bars, counts):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                           f'{count}', ha='center', va='bottom')

                plt.xticks(rotation=45)
                st.pyplot(fig)

        else:
            st.warning("😔 Лица не обнаружены")
            st.info("Попробуйте сделать фото с хорошим освещением и четким изображением лица")

            st.write("Анализ всего изображения...")
            probs = predict_on_bgr_image(image_bgr, model, device, img_size)
            emotion_idx = np.argmax(probs)
            emotion = CLASSES[emotion_idx]
            confidence = float(probs[emotion_idx])

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Предсказанная эмоция", emotion)
                st.metric("Уверенность", f"{confidence:.2%}")

            with col2:
                st.write("**Все вероятности:**")
                for i, prob in enumerate(probs):
                    st.write(f"- {CLASSES[i]}: {prob:.2%}")

            st.image(image, caption="Исходное изображение", use_container_width=True)

    except Exception as e:
        st.error(f"❌ Ошибка при обработке изображения: {e}")
        import traceback
        st.error(f"Детали ошибки: {traceback.format_exc()}")

def create_complete_zip_report(all_results, df, winner, all_test_infos, test_path, img_size, batch_size, device):

    import zipfile
    import io

    print(f"DEBUG: Начинаем создание ZIP с {len(all_results)} моделей")
    print(f"DEBUG: DF shape: {df.shape}")

    try:

        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            print("DEBUG: ZIP файл создан, начинаем добавлять данные")

            print("DEBUG: Создаем CSV...")
            detailed_csv_data = []
            headers = [
                'Rank', 'Model', 'Accuracy', 'F1_Macro', 'Avg_ROC_AUC',
                'ROC_AUC_angry', 'ROC_AUC_disgust', 'ROC_AUC_fear', 'ROC_AUC_happy',
                'ROC_AUC_neutral', 'ROC_AUC_sad', 'ROC_AUC_surprise',
                'Precision_angry', 'Recall_angry', 'F1_angry',
                'Precision_disgust', 'Recall_disgust', 'F1_disgust',
                'Precision_fear', 'Recall_fear', 'F1_fear',
                'Precision_happy', 'Recall_happy', 'F1_happy',
                'Precision_neutral', 'Recall_neutral', 'F1_neutral',
                'Precision_sad', 'Recall_sad', 'F1_sad',
                'Precision_surprise', 'Recall_surprise', 'F1_surprise',
                'True_angry', 'True_disgust', 'True_fear', 'True_happy',
                'True_neutral', 'True_sad', 'True_surprise',
                'Pred_angry', 'Pred_disgust', 'Pred_fear', 'Pred_happy',
                'Pred_neutral', 'Pred_sad', 'Pred_surprise',
                'Support_angry', 'Support_disgust', 'Support_fear', 'Support_happy',
                'Support_neutral', 'Support_sad', 'Support_surprise',
                'Error_Status'
            ]
            detailed_csv_data.append(','.join(headers))

            print(f"DEBUG: Обрабатываем {len(df)} моделей для CSV")
            for idx, row in df.iterrows():
                if idx % 5 == 0:
                    print(f"DEBUG: Обработка модели {idx+1}/{len(df)}: {row['model']}")

                model_result = all_results[idx]
                model_name = row['model']

                base_data = [
                    str(row['rank']),
                    f'"{model_name}"',
                    f"{row['accuracy']:.6f}",
                    f"{row['f1_macro']:.6f}",
                    f"{row['roc_auc_avg']:.6f}"
                ]

                roc_auc_data = []
                if 'roc_auc' in model_result and isinstance(model_result['roc_auc'], dict):
                    for cls in CLASSES:
                        roc_auc_data.append(f"{model_result['roc_auc'].get(cls, 0):.6f}")
                else:
                    roc_auc_data = ['0.000000'] * len(CLASSES)

                precision_data = []
                recall_data = []
                f1_data = []
                support_data = []

                if 'report' in model_result and model_result['report']:
                    report_lines = model_result['report'].split('\n')
                    for line in report_lines:
                        if any(cls in line for cls in CLASSES):
                            parts = line.split()
                            if len(parts) >= 4:
                                try:
                                    cls_name = None
                                    for cls in CLASSES:
                                        if line.strip().startswith(cls):
                                            cls_name = cls
                                            break

                                    if cls_name:
                                        precision = float(parts[-4])
                                        recall = float(parts[-3])
                                        f1 = float(parts[-2])
                                        support = int(parts[-1])

                                        precision_data.append(f"{precision:.6f}")
                                        recall_data.append(f"{recall:.6f}")
                                        f1_data.append(f"{f1:.6f}")
                                        support_data.append(str(support))
                                except (ValueError, IndexError):
                                    precision_data.append("0.000000")
                                    recall_data.append("0.000000")
                                    f1_data.append("0.000000")
                                    support_data.append("0")

                while len(precision_data) < len(CLASSES):
                    precision_data.append("0.000000")
                    recall_data.append("0.000000")
                    f1_data.append("0.000000")
                    support_data.append("0")

                confusion_data = []
                if 'confusion' in model_result and model_result['confusion'] is not None:
                    cm = model_result['confusion']
                    if isinstance(cm, np.ndarray):
                        for i in range(len(CLASSES)):
                            confusion_data.append(str(int(cm[i].sum())))
                        for j in range(len(CLASSES)):
                            confusion_data.append(str(int(cm[:, j].sum())))
                else:
                    confusion_data = ['0'] * (len(CLASSES) * 2)

                error_status = model_result.get('error', 'Success')
                if error_status:
                    error_status = f'"{error_status}"'
                else:
                    error_status = 'Success'

                full_row = base_data + roc_auc_data + precision_data + recall_data + f1_data + confusion_data + support_data + [error_status]
                detailed_csv_data.append(','.join(full_row))

            detailed_csv_text = '\n'.join(detailed_csv_data)
            zip_file.writestr("models_super_detailed_report.csv", detailed_csv_text)
            print("DEBUG: CSV добавлен в ZIP")

            print("DEBUG: Добавляем ROC кривые...")
            for idx, row in df.iterrows():
                model_result = all_results[idx]
                model_name = row['model'].replace('.pth', '')

                if 'fig_roc' in model_result and model_result['fig_roc']:
                    img_buffer = io.BytesIO()
                    model_result['fig_roc'].savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
                    img_buffer.seek(0)

                    zip_file.writestr(f"roc_curves/roc_curve_{model_name}.png", img_buffer.getvalue())

                    metadata = f
                    if 'roc_auc' in model_result and isinstance(model_result['roc_auc'], dict):
                        for cls, auc_val in model_result['roc_auc'].items():
                            metadata += f"{cls}: {auc_val:.6f}\n"

                    zip_file.writestr(f"roc_curves/roc_metadata_{model_name}.txt", metadata)

            readme_content = f
            zip_file.writestr("README.txt", readme_content)
            print("DEBUG: README добавлен в ZIP")

        zip_buffer.seek(0)
        result = zip_buffer.getvalue()
        print(f"DEBUG: ZIP успешно создан, размер: {len(result)} байт")
        return result

    except Exception as e:
        print(f"DEBUG: Ошибка при создании ZIP: {e}")
        import traceback
        print(f"DEBUG: Traceback: {traceback.format_exc()}")
        raise e

if __name__ == '__main__':
    main()
