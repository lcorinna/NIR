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
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report
from datetime import datetime
import pytz

# =============================
# 1. Параметры модели и обучения
# =============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4
EPOCHS = 50
LEARNING_RATE = 0.0005
DATASET_PATH = r"E:/MAI/NIR/Faster_R-CNN/dataset/"

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

        self.image_info = {img["id"]: img["file_name"] for img in self.coco_data["images"]}
        self.annotations = {ann["image_id"]: ann for ann in self.coco_data["annotations"]}
        self.class_names = {cat["id"]: cat["name"] for cat in self.coco_data["categories"]}
        
        # Добавляем "фон" (Background) как класс 0
        self.class_names[0] = "background"

    def __getitem__(self, idx):
        img_name = self.image_info[idx]
        img_path = os.path.join(self.root, img_name)

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = F.to_tensor(img)

        ann = self.annotations.get(idx, {"bbox": [], "category_id": []})
        boxes = torch.tensor([ann["bbox"]], dtype=torch.float32) if ann["bbox"] else torch.zeros((0, 4))
        labels = torch.tensor([ann["category_id"]], dtype=torch.int64) if ann["category_id"] else torch.zeros((0,), dtype=torch.int64)

        # Преобразуем COCO bbox [x, y, w, h] → [x_min, y_min, x_max, y_max]
        if len(boxes) > 0:
            boxes[:, 2] += boxes[:, 0]
            boxes[:, 3] += boxes[:, 1]
            boxes[:, 2] = torch.clamp(boxes[:, 2], min=boxes[:, 0] + 1)
            boxes[:, 3] = torch.clamp(boxes[:, 3], min=boxes[:, 1] + 1)

        target = {"boxes": boxes, "labels": labels}
        return img, target

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

    num_classes = len(train_dataset.class_names)
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

    torch.save(model.state_dict(), "faster_rcnn.pth")
    print("✅ Модель сохранена!")

    # =============================
    # 7. Оценка модели
    # =============================
    def evaluate_model(model, test_loader, class_names):
        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for imgs, targets in test_loader:
                imgs = list(img.to(DEVICE) for img in imgs)
                predictions = model(imgs)

                for pred, target in zip(predictions, targets):
                    pred_labels = pred["labels"].cpu().numpy().tolist()
                    target_labels = target["labels"].cpu().numpy().tolist()

                    all_preds.extend(pred_labels)
                    all_targets.extend(target_labels)

        if len(all_preds) == 0 or len(all_targets) == 0:
            print("⚠️ Нет предсказаний или аннотаций!")
            return

        print("\n📊 **Метрики по каждому классу:**")
        precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, zero_division=0)
        accuracy = accuracy_score(all_targets, all_preds)

        for i, class_name in enumerate(sorted(class_names.values())):
            print(f"{class_name}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1-score={f1[i]:.4f}")

        print(f"\n📊 **Общая точность (Accuracy): {accuracy:.4f}**")

    # 🔥 Оценка модели
    evaluate_model(model, test_loader, train_dataset.class_names)
