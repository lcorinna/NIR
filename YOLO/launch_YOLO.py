import torch
import cv2
import numpy as np
from ultralytics import YOLO
import sys

# Укажи путь к файлу модели
MODEL_PATH = r"E:\MAI\NIR\runs\diploma_training\experiment_12\weights\last.pt"
# Укажи путь к изображению
IMAGE_PATH = r"E:\MAI\NIR\изображения для проверки\image4.jpg"

# Определение устройства (CPU или GPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Загрузка модели...")

# Загружаем модель
try:
    model = YOLO(MODEL_PATH)
    print("Модель загружена успешно!")
except Exception as e:
    print(f"Ошибка загрузки модели: {e}")
    sys.exit(1)

# Запуск предсказания
print("Классификация изображения...")
try:
    results = model(IMAGE_PATH, device=DEVICE, save=True)
except Exception as e:
    print(f"Ошибка предсказания: {e}")
    sys.exit(1)

# Обработка результатов
if results and len(results[0].boxes):
    img = cv2.imread(IMAGE_PATH)
    class_labels = {0: "Dent", 1: "Fastener Damage", 2: "Rupture"}
    
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        label = int(box.cls.cpu().numpy())
        confidence = float(box.conf.cpu().numpy()) * 100
        class_name = class_labels.get(label, "Неизвестный класс")
        
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{class_name} {confidence:.2f}%", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Показываем изображение с боксами
    cv2.imshow("Detected Damage", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(f'Предсказанный класс: {class_name}, уверенность: {confidence:.2f}%')
else:
    print("Модель не обнаружила повреждений на изображении.")
