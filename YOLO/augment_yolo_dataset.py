import os
import cv2
import glob
import random
import shutil
import albumentations as A
import numpy as np
from tqdm import tqdm

# === Настройки ===
data_dir = "datasets"
target_counts = {
    'train': 2500,
    'valid': 300,
    'test': 300
}
class_names = ['Dent', 'Fastener Damage', 'Rupture']
class_ids = [0, 1, 2]

# === Аугментации ===
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=30, p=0.7),
    A.RandomBrightnessContrast(p=0.5),
    A.RandomScale(scale_limit=0.2, p=0.5),
], keypoint_params=A.KeypointParams(format='yolo', remove_invisible=False))

# === Обход по train/valid/test ===
for split in ['train', 'valid', 'test']:
    print(f"\n🔍 Обработка: {split}")
    images_dir = os.path.join(data_dir, split, 'images')
    labels_dir = os.path.join(data_dir, split, 'labels')

    # Соберём файлы по классам
    class_to_files = {i: [] for i in class_ids}
    all_txt = glob.glob(os.path.join(labels_dir, '*.txt'))

    for txt_file in all_txt:
        with open(txt_file, 'r') as f:
            lines = f.readlines()
        classes_in_file = set([int(line.split()[0]) for line in lines])
        for class_id in classes_in_file:
            if class_id in class_to_files:
                class_to_files[class_id].append(txt_file)

    # Для каждого класса
    for class_id in class_ids:
        existing = len(class_to_files[class_id])
        needed = target_counts[split] - existing
        print(f"\n✅ Класс {class_id} ({class_names[class_id]}): есть {existing}, нужно добавить {needed}")

        if needed <= 0:
            continue

        i = 0
        while i < needed:
            src_label = random.choice(class_to_files[class_id])
            src_image = src_label.replace('labels', 'images').replace('.txt', '.jpg')

            if not os.path.exists(src_image):
                continue

            # Загрузим исходные данные
            image = cv2.imread(src_image)
            height, width = image.shape[:2]
            with open(src_label, 'r') as f:
                lines = f.readlines()

            # Соберём аннотации текущего класса
            keypoints = []
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                cid = int(parts[0])
                if len(parts) == 5:
                    x, y, w, h = map(float, parts[1:])
                    keypoints.append((x, y, w, h, cid))
                else:
                    # Пропустим полигоны для упрощения в этой версии
                    continue

            # Только bbox
            bboxes = [(x, y, w, h) for x, y, w, h, cid in keypoints if cid == class_id]
            if not bboxes:
                continue

            try:
                transformed = transform(image=image, bboxes=bboxes)
            except Exception as e:
                continue

            aug_image = transformed['image']
            aug_bboxes = transformed['bboxes']

            if not aug_bboxes:
                continue

            # Сохраняем
            base_name = f"aug_{i}_{os.path.basename(src_image)}"
            aug_img_path = os.path.join(images_dir, base_name)
            aug_txt_path = os.path.join(labels_dir, base_name.replace('.jpg', '.txt'))
            cv2.imwrite(aug_img_path, aug_image)

            with open(aug_txt_path, 'w') as f:
                for bbox in aug_bboxes:
                    x, y, w, h = bbox
                    f.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

            i += 1

print("\n✅ Аугментация завершена!")
