import torch
from ultralytics import YOLO
import datetime

if __name__ == '__main__':
    # Установление устройства для обучения
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Устройство для обучения: {device}")

    # Загрузка предобученной модели YOLOv11
    model = YOLO("yolo11n.pt")  # Загрузка модели

    # Установление уникального имени эксперимента
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"experiment_{timestamp}"

    # Установление параметров обучения
    results = model.train(
        data="E://MAI//NIR//datasets//data.yaml",       # Путь к файлу data.yaml
        epochs=50,                      # Количество эпох
        imgsz=640,                      # Размер входных изображений
        batch=16,                       # Размер батча
        device=device,                  # Использование GPU или CPU
        save=True,                      # Сохранение результатов
        project="runs/diploma_training",# Папка для сохранения результатов
        name="experiment_name",         # Название эксперимента
        verbose=True,                   # Подробные логи в процессе
        plots=True                      # Автоматическое создание графиков
    )

    print("Результаты тестирования сохранены в папке runs/predict")
