import torch
import torchvision.transforms as T
import numpy as np
import cv2
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import sys

# Путь к файлу модели
MODEL_PATH = "faster_rcnn.pth"
# Путь к изображению, которое нужно классифицировать
IMAGE_PATH = "E:\\MAI\\NIR\\изображения для проверки\\image4.jpg"

# Определение устройства (CPU или GPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Загрузка модели...")

# Загружаем модель
def load_faster_rcnn_model(model_path):
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=4)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

try:
    model = load_faster_rcnn_model(MODEL_PATH)
    print("Модель загружена успешно!")
except Exception as e:
    print(f"Ошибка загрузки модели: {e}")
    sys.exit(1)

# Загружаем изображение
def process_image(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
        img_np = np.array(img)
        transform = T.Compose([T.ToTensor()])
        img_tensor = transform(img).to(DEVICE)
        return img_np, img_tensor
    except Exception as e:
        print(f"Ошибка загрузки изображения: {e}")
        sys.exit(1)

img_np, img_tensor = process_image(IMAGE_PATH)

# Получаем предсказание
print("Классификация изображения...")
with torch.no_grad():
    predictions = model([img_tensor])[0]

# Определяем класс с максимальной вероятностью
class_labels = {1: "Dent", 2: "Fastener Damage", 3: "Rupture"}
if len(predictions["scores"]) > 0:
    best_idx = predictions["scores"].argmax().item()
    predicted_class = class_labels.get(predictions["labels"][best_idx].item(), "Неизвестный класс")
    confidence = predictions["scores"][best_idx].item() * 100
    print(f'Предсказанный класс: {predicted_class}, уверенность: {confidence:.2f}%')
    
    # Отображение предсказаний
    for box, label, score in zip(predictions["boxes"].cpu().numpy(), predictions["labels"].cpu().numpy(), predictions["scores"].cpu().numpy()):
        if score < 0.3:
            continue
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_np, f"{class_labels.get(label, 'Unknown')} {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Показываем изображение с боксами
    cv2.imshow("Detected Damage", cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Модель не обнаружила повреждений на изображении.")
