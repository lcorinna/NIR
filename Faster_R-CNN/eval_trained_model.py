import torch
import json
import os
import numpy as np
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as F
from sklearn.metrics import classification_report
from collections import defaultdict
from PIL import Image


# ============== Параметры ==============
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_PATH = r"E:/MAI/NIR/Faster_R-CNN/dataset/"
MODEL_PATH = "faster_rcnn.pth"

# ============== Класс для загрузки тестовых данных ==============
class CustomDataset(Dataset):
    def __init__(self, root, split="test"):
        self.root = os.path.join(root, split)
        ann_file = os.path.join(self.root, "_annotations.coco.json")

        if not os.path.exists(ann_file):
            raise FileNotFoundError(f"❌ Файл аннотаций {ann_file} не найден!")

        with open(ann_file, "r") as f:
            self.coco_data = json.load(f)

        # ❗ Удаляем суперкласс "-dents-leaks-ruptures-other" (id: 0)
        self.class_names = {cat["id"]: cat["name"] for cat in self.coco_data["categories"] if cat["id"] != 0}

        self.image_info = {img["id"]: img["file_name"] for img in self.coco_data["images"]}
        self.annotations = defaultdict(list)

        for ann in self.coco_data["annotations"]:
            if ann["category_id"] != 0:
                self.annotations[ann["image_id"]].append(ann)

    def __getitem__(self, idx):
        img_name = self.image_info[idx]
        img_path = os.path.join(self.root, img_name)

        img = F.to_tensor(Image.open(img_path).convert("RGB"))

        anns = self.annotations[idx]
        boxes = torch.tensor([ann["bbox"] for ann in anns], dtype=torch.float32) if anns else torch.zeros((0, 4))
        labels = torch.tensor([ann["category_id"] for ann in anns], dtype=torch.int64) if anns else torch.zeros((0,), dtype=torch.int64)

        if len(boxes) > 0:
            boxes[:, 2] += boxes[:, 0]  # x_max = x_min + width
            boxes[:, 3] += boxes[:, 1]  # y_max = y_min + height

        return img, {"boxes": boxes, "labels": labels}

    def __len__(self):
        return len(self.image_info)

# ============== Функция загрузки обученной модели ==============
def load_faster_rcnn_model(num_classes, model_path):
    model = fasterrcnn_resnet50_fpn(weights=None)  # Загружаем без предобученных весов
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()
    return model

# ============== Оценка модели ==============
def evaluate_model(model, test_loader, class_names):
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for imgs, targets in test_loader:
            imgs = list(img.to(DEVICE) for img in imgs)
            predictions = model(imgs)

            for pred, target in zip(predictions, targets):
                pred_labels = pred["labels"].cpu().numpy().tolist()
                target_labels = target["labels"].cpu().numpy().tolist()

                # ✅ Корректируем предсказания, чтобы длины совпадали
                max_len = max(len(pred_labels), len(target_labels))
                pred_labels += [0] * (max_len - len(pred_labels))
                target_labels += [0] * (max_len - len(target_labels))

                all_preds.extend(pred_labels)
                all_targets.extend(target_labels)

    if len(all_preds) == 0 or len(all_targets) == 0:
        print("⚠️ Нет предсказаний или аннотаций! Проверьте данные.")
        return

    # ✅ Выводим метрики по каждому классу
    print("\n📊 **Метрики по каждому классу:**")
    print(classification_report(
        all_targets, all_preds, labels=sorted(test_dataset.class_names.keys()), target_names=list(test_dataset.class_names.values()), zero_division=1
    ))

# ============== Запуск оценки ==============
if __name__ == '__main__':
    test_dataset = CustomDataset(DATASET_PATH, split="test")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=lambda batch: tuple(zip(*batch)))

    num_classes = len(test_dataset.class_names) + 1
    model = load_faster_rcnn_model(num_classes, MODEL_PATH)

    evaluate_model(model, test_loader, test_dataset.class_names)
