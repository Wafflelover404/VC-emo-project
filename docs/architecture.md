# Схема взаимодействия компонентов

## Архитектура системы

```mermaid
graph TB
    subgraph "Уровень данных"
        A[train/]
        B[test/]
        C[fer2013.csv]
    end
    
    subgraph "Уровень подготовки данных"
        D[prepare_dataset.py]
        E[split_dataset.py]
    end
    
    subgraph "Уровень обучения"
        F[train_model.py]
        G[metrics.py]
    end
    
    subgraph "Уровень модели"
        H[wafflelover404_emotion_model.pth]
        I[wafflelover404_emotion_model_best.pth]
    end
    
    subgraph "Уровень инференса"
        J[camera_inference.py]
        K[streamlit_app.py]
        L[camera_test.py]
    end
    
    subgraph "Уровень результатов"
        M[metrics/]
        N[logs/]
    end
    
    subgraph "Внешние зависимости"
        O[OpenCV]
        P[PyTorch]
        Q[Streamlit]
        R[scikit-learn]
    end
    
    C --> D
    D --> A
    D --> B
    A --> F
    B --> F
    F --> H
    F --> I
    F --> G
    G --> M
    H --> J
    H --> K
    I --> J
    I --> K
    J --> O
    K --> Q
    F --> P
    G --> R
    
    style A fill:#e1f5fe
    style B fill:#e1f5fe
    style H fill:#c8e6c9
    style I fill:#c8e6c9
    style M fill:#fff3e0
```

## Компоненты и их взаимодействие

### 1. Модули подготовки данных
- **prepare_dataset.py**: Конвертирует CSV в изображения
- **split_dataset.py**: Разделяет данные на train/test

### 2. Модуль обучения
- **train_model.py**: Основной скрипт обучения
- **metrics.py**: Утилиты для расчета метрик

### 3. Модули инференса
- **camera_inference.py**: Работа с камерой через OpenCV
- **streamlit_app.py**: Веб-интерфейс
- **camera_test.py**: Тестирование камеры

### 4. Потоки данных

#### Поток обучения:
```
CSV → prepare_dataset.py → train/test → train_model.py → model.pth → metrics/
```

#### Поток инференса:
```
model.pth → camera_inference.py/streamlit_app.py → камера/файл → результаты
```

## Зависимости

### Основные:
- **PyTorch**: Нейросеть и обучение
- **OpenCV**: Работа с изображениями и камерой
- **Streamlit**: Веб-интерфейс

### Вспомогательные:
- **scikit-learn**: Метрики и ROC-кривые
- **matplotlib/seaborn**: Визуализация
- **Pillow**: Обработка изображений
- **numpy**: Вычисления

## Запуск

### Быстрый запуск:
```bash
./start.sh
```

### Поэтапный запуск:
```bash
# 1. Подготовка данных
python prepare_dataset.py

# 2. Обучение
python train_model.py

# 3. Инференс
python camera_inference.py
# или
streamlit run streamlit_app.py
```

## Хранение результатов

- **Модели**: `*.pth` файлы
- **Метрики**: папка `metrics/`
- **Логи**: папка `logs/`
