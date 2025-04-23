import os
import cv2
import random
import pandas as pd
import numpy as np
from glob import glob

# === 🔧 НАСТРОЙКА ПОЛЬЗОВАТЕЛЕМ ===
class_name = 'Rupture'               # ← просто меняй это значение [Dent, Fastener Damage, Rupture]
target_per_class = 300            # ← сколько всего изображений должно быть после аугментации
dataset_split = 'valid'            # ← 'train', 'valid', 'test'
# =================================

# Пути
base_path = os.path.abspath(f'../data/{dataset_split}')
annotations_path = os.path.join(base_path, '_annotations.csv')
class_dir = os.path.join(base_path, class_name)

# Загружаем аннотации
df = pd.read_csv(annotations_path)

# Считаем, сколько изображений есть
images = glob(os.path.join(class_dir, '*.jpg'))
current_count = len(images)
needed = target_per_class - current_count

print(f'🔧 Класс: {class_name}, есть: {current_count}, нужно добавить: {needed}')

if needed <= 0:
    print(f'✅ Уже достаточно изображений для класса {class_name}')
    exit()

# Функция аугментации
def augment_image(img):
    rows, cols = img.shape[:2]

    # Геометрия
    if random.random() > 0.5:
        img = cv2.flip(img, 1)
    if random.random() > 0.5:
        img = cv2.flip(img, 0)
    angle = random.uniform(-30, 30)
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    img = cv2.warpAffine(img, M, (cols, rows))

    # Освещение
    alpha = random.uniform(0.8, 1.2)
    beta = random.randint(-20, 20)
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    return img

# Генерация новых изображений
new_records = []
i = 0
while i < needed:
    src_img_path = random.choice(images)
    img = cv2.imread(src_img_path)

    if img is None:
        continue

    aug_img = augment_image(img)
    new_name = f'aug_{i}_{os.path.basename(src_img_path)}'
    new_path = os.path.join(class_dir, new_name)
    cv2.imwrite(new_path, aug_img)

    new_records.append({'filename': f'{class_name}/{new_name}', 'class': class_name})
    i += 1

# Обновляем CSV
df_aug = pd.DataFrame(new_records)
df_combined = pd.concat([df, df_aug], ignore_index=True)
df_combined.to_csv(annotations_path, index=False)

print(f"✅ Добавлено {len(new_records)} новых изображений в класс '{class_name}' и обновлён _annotations.csv")
