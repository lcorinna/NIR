import torch  
import torchvision
import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from sklearn.metrics import classification_report
from collections import defaultdict
from datetime import datetime
import pytz
from PIL import Image

# =============================
# 1. Параметры модели и обучения
# =============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4
EPOCHS = 50
LEARNING_RATE = 0.0005
DATASET_PATH = r"E:/MAI/NIR/Faster_R-CNN/dataset/"
MODEL_PATH = "faster_rcnn.pth"

# =============================
# 2. Проверка пути к dataset
# =============================
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"❌ Ошибка: Папка {DATASET_PATH} не найдена!")

# =============================
# 3. Кастомный датасет COCO
# =============================
class CustomDataset(Dataset):
    def __init__(self, root, split="train"):
        self.root = os.path.join(root, split)
        ann_file = os.path.join(self.root, "_annotations.coco.json")

        if not os.path.exists(ann_file):
            raise FileNotFoundError(f"❌ Файл аннотаций {ann_file} не найден!")

        with open(ann_file, "r") as f:
            self.coco_data = json.load(f)

        self.class_names = {cat["id"]: cat["name"] for cat in self.coco_data["categories"]}
        self.image_info = {img["id"]: img["file_name"] for img in self.coco_data["images"]}
        self.annotations = defaultdict(list)

        for ann in self.coco_data["annotations"]:
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

# =============================
# 4. Основной блок (Windows Fix)
# =============================
if __name__ == '__main__':
    train_dataset = CustomDataset(DATASET_PATH, split="train")
    test_dataset = CustomDataset(DATASET_PATH, split="test")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=lambda batch: tuple(zip(*batch)))
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=lambda batch: tuple(zip(*batch)))

    # =============================
    # 5. Определяем модель Faster R-CNN
    # =============================
    def get_faster_rcnn_model(num_classes):
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn(weights=weights)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model

    num_classes = len(train_dataset.class_names) + 1
    model = get_faster_rcnn_model(num_classes).to(DEVICE)

    # =============================
    # 6. Обучение модели
    # =============================
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    print("🚀 Начинаем обучение...")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        moscow_time = datetime.now(pytz.timezone('Europe/Moscow')).strftime("%H:%M:%S")
        print(f"🔥 Начало эпохи {epoch+1}/{EPOCHS} | Время (Москва): {moscow_time}")

        for imgs, targets in train_loader:
            imgs = list(img.to(DEVICE) for img in imgs)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            loss_dict = model(imgs, targets)
            loss = sum(loss for loss in loss_dict.values())

            if torch.isnan(loss):
                print(f"⚠️ Loss NaN в эпохе {epoch+1}")
                break

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        lr_scheduler.step()
        print(f"✅ Эпоха {epoch+1} завершена. | Время (Москва): {moscow_time} | Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print("✅ Модель сохранена!")

    # =============================
    # 7. Оценка модели
    # =============================
    def load_faster_rcnn_model(num_classes, model_path):
        model = fasterrcnn_resnet50_fpn(weights=None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        return model

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

                    max_len = max(len(pred_labels), len(target_labels))
                    pred_labels += [0] * (max_len - len(pred_labels))
                    target_labels += [0] * (max_len - len(target_labels))

                    all_preds.extend(pred_labels)
                    all_targets.extend(target_labels)

        if len(all_preds) == 0 or len(all_targets) == 0:
            print("⚠️ Нет предсказаний или аннотаций! Проверьте данные.")
            return

        print("\n📊 **Метрики по каждому классу:**")
        print(classification_report(
            all_targets, all_preds, labels=sorted(test_dataset.class_names.keys()), target_names=list(test_dataset.class_names.values()), zero_division=1
        ))

    print("🔍 Запуск оценки модели...")
    model = load_faster_rcnn_model(num_classes, MODEL_PATH)
    evaluate_model(model, test_loader, test_dataset.class_names)
