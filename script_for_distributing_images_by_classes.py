import pandas as pd
import os
import shutil

# Параметры
data_dir = 'data'
src_images_dir = os.path.join(data_dir, 'images')

# Проверка наличия исходной директории с изображениями
if not os.path.exists(src_images_dir):
    print(f"Source images directory {src_images_dir} does not exist! Please place your images in this directory.")
    exit()

# Функция для распределения изображений по классам
def distribute_images(data_dir, dataset_type):
    class_labels_file = os.path.join(data_dir, dataset_type, '_classes.csv')
    if not os.path.exists(class_labels_file):
        print(f"Class labels file {class_labels_file} does not exist! Please check your files.")
        return

    labels_df = pd.read_csv(class_labels_file)
    labels_df.columns = labels_df.columns.str.strip()  # Удаление пробелов из названий колонок
    labels_df['label'] = labels_df[['Dent', 'Fastener Damage', 'Rupture']].idxmax(axis=1)

    # Используем правильное имя колонки с именами изображений
    for _, row in labels_df.iterrows():
        image_name = row['filename']  # замените 'filename' на правильное имя колонки
        class_label = row['label']
        src_path = os.path.join(src_images_dir, image_name)
        dest_dir = os.path.join(data_dir, dataset_type, class_label)
        dest_path = os.path.join(dest_dir, image_name)
        
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        
        if os.path.exists(src_path):
            shutil.copy(src_path, dest_path)
        else:
            print(f"Image {src_path} not found!")

# Распределение изображений для train, valid, test
for dataset_type in ['train', 'valid', 'test']:
    distribute_images(data_dir, dataset_type)

# Проверка структуры директорий и наличия изображений после распределения
def check_directory_structure(base_dir):
    classes = ['Dent', 'Fastener Damage', 'Rupture']
    for dataset_type in ['train', 'valid', 'test']:
        dataset_dir = os.path.join(base_dir, dataset_type)
        print(f"Checking {dataset_type} directory: {dataset_dir}")
        for class_name in classes:
            class_dir = os.path.join(dataset_dir, class_name)
            if os.path.exists(class_dir):
                images = os.listdir(class_dir)
                print(f"Class directory {class_dir} found with {len(images)} images.")
                if images:
                    print(f"Sample images in {class_dir}: {images[:5]}")  # Показать первые 5 изображений
            else:
                print(f"Class directory {class_dir} does not exist!")

# Проверка структуры директорий после распределения
check_directory_structure(data_dir)
