import torch
import torchvision
import cv2
import matplotlib.pyplot as plt
import numpy as np
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F

# =============================
# 1. Параметры
# =============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "faster_rcnn.pth"  # Путь к сохраненной модели
TEST_IMAGE_PATH = "E:/MAI/NIR/Faster-R-CNN/dataset/test/example.jpg"  # Замените на свой путь

# =============================
# 2. Функция загрузки модели
# =============================
def load_model(num_classes, model_path):
    """
    Загружает обученную модель Faster R-CNN и подготавливает её для инференса.
    
    :param num_classes: Количество классов
    :param model_path: Путь к файлу модели (faster_rcnn.pth)
    :return: Загрузка модели на GPU/CPU
    """
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT  # Используем предобученные веса
    model = fasterrcnn_resnet50_fpn(weights=weights)
    
    # Меняем выходной слой под наши классы
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Загружаем веса модели
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()  # Переключаем в режим инференса
    print("✅ Модель загружена!")
    return model

# =============================
# 3. Загрузка тестового изображения
# =============================
def load_image(image_path):
    """
    Загружает изображение и конвертирует его в тензор для инференса.
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Конвертация в RGB
    tensor = F.to_tensor(image).to(DEVICE).unsqueeze(0)  # Преобразуем в PyTorch-тензор
    return image, tensor

# =============================
# 4. Запуск модели на тестовом изображении
# =============================
def run_inference(model, image_tensor, confidence_threshold=0.5):
    """
    Запускает инференс на изображении и фильтрует результаты по confidence score.
    
    :param model: Загруженная модель
    :param image_tensor: Тензор изображения
    :param confidence_threshold: Порог уверенности предсказаний
    :return: Bounding boxes, labels, scores
    """
    with torch.no_grad():
        predictions = model(image_tensor)

    boxes = predictions[0]["boxes"].cpu().numpy()
    labels = predictions[0]["labels"].cpu().numpy()
    scores = predictions[0]["scores"].cpu().numpy()

    # Фильтрация объектов по confidence threshold
    valid_indices = scores > confidence_threshold
    return boxes[valid_indices], labels[valid_indices], scores[valid_indices]

# =============================
# 5. Отображение предсказаний
# =============================
def visualize_results(image, boxes, labels, scores):
    """
    Рисует предсказанные bounding boxes на изображении.
    """
    fig, ax = plt.subplots(1, figsize=(10, 6))
    ax.imshow(image)

    for i, (box, score) in enumerate(zip(boxes, scores)):
        x_min, y_min, x_max, y_max = box
        rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fill=False, color="red", linewidth=2)
        ax.add_patch(rect)
        ax.text(x_min, y_min - 5, f"Class: {labels[i]} | {score:.2f}", color="red", fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

    plt.show()

# =============================
# 6. Основной код
# =============================
if __name__ == "__main__":
    # 1️⃣ Загрузка модели
    num_classes = 4  # Укажите число классов в датасете + 1 (фон)
    model = load_model(num_classes, MODEL_PATH)

    # 2️⃣ Загрузка изображения
    image, image_tensor = load_image(TEST_IMAGE_PATH)

    # 3️⃣ Запуск инференса
    boxes, labels, scores = run_inference(model, image_tensor)

    # 4️⃣ Визуализация
    visualize_results(image, boxes, labels, scores)
