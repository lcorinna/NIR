import os
import json
import random
import shutil
from tqdm import tqdm
import albumentations as A
import cv2
import numpy as np

# === Настройки ===
data_dir = 'dataset'  # корневая папка
splits = {
    'train': 2500,
    'valid': 300,
    'test': 300
}

class_names = ['Dent', 'Fastener Damage', 'Rupture']  # порядок строго как в аннотации
class_ids = [1, 2, 3]  # COCO обычно использует id начиная с 1

# Аугментации
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=30, p=0.7),
    A.RandomBrightnessContrast(p=0.5),
    A.RandomScale(scale_limit=0.2, p=0.5),
], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))

# === Функция для генерации уникальных ID ===
def get_next_id(existing_ids):
    current = max(existing_ids) + 1 if existing_ids else 1
    while True:
        yield current
        current += 1

# === Обработка каждого сплита ===
for split, target_per_class in splits.items():
    print(f"\n📁 Обработка: {split}")
    split_dir = os.path.join(data_dir, split)
    images_dir = split_dir
    json_path = os.path.join(split_dir, '_annotations.coco.json')

    with open(json_path, 'r') as f:
        coco = json.load(f)

    images = coco['images']
    annotations = coco['annotations']
    categories = coco['categories']

    class_to_images = {cid: [] for cid in class_ids}
    image_id_to_image = {img['id']: img for img in images}
    image_id_to_anns = {}

    for ann in annotations:
        image_id = ann['image_id']
        image_id_to_anns.setdefault(image_id, []).append(ann)
        if ann['category_id'] in class_ids:
            class_to_images[ann['category_id']].append(image_id)

    # Генераторы ID
    image_ids_used = {img['id'] for img in images}
    ann_ids_used = {ann['id'] for ann in annotations}
    new_image_id = get_next_id(image_ids_used)
    new_ann_id = get_next_id(ann_ids_used)

    for class_id in class_ids:
        current_ids = set(class_to_images[class_id])
        current_count = len(current_ids)
        needed = target_per_class - current_count
        print(f"\n✅ Класс {class_id} ({class_names[class_id-1]}): есть {current_count}, нужно добавить {needed}")

        if needed <= 0:
            continue

        i = 0
        while i < needed:
            base_img_id = random.choice(list(current_ids))
            base_img_info = image_id_to_image[base_img_id]
            base_img_path = os.path.join(images_dir, base_img_info['file_name'])
            img = cv2.imread(base_img_path)
            if img is None:
                continue

            height, width = img.shape[:2]
            anns = [a for a in image_id_to_anns[base_img_id] if a['category_id'] == class_id]
            bboxes = [a['bbox'] for a in anns]
            category_ids = [class_id] * len(bboxes)

            if not bboxes:
                continue

            try:
                transformed = transform(image=img, bboxes=bboxes, category_ids=category_ids)
            except Exception:
                continue

            aug_img = transformed['image']
            aug_bboxes = transformed['bboxes']
            aug_cat_ids = transformed['category_ids']

            if not aug_bboxes:
                continue

            new_img_filename = f"aug_{i}_{base_img_info['file_name']}"
            new_img_path = os.path.join(images_dir, new_img_filename)
            cv2.imwrite(new_img_path, aug_img)

            img_id = next(new_image_id)
            coco['images'].append({
                'id': img_id,
                'file_name': new_img_filename,
                'width': width,
                'height': height
            })

            for bbox, cat_id in zip(aug_bboxes, aug_cat_ids):
                ann_id = next(new_ann_id)
                coco['annotations'].append({
                    'id': ann_id,
                    'image_id': img_id,
                    'category_id': cat_id,
                    'bbox': [float(x) for x in bbox],
                    'area': float(bbox[2] * bbox[3]),
                    'iscrowd': 0
                })

            i += 1

    # Сохраняем обновленный JSON
    with open(json_path, 'w') as f:
        json.dump(coco, f, indent=2)

print("\n✅ Аугментация COCO завершена!")
