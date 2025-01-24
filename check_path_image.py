import os

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

# Путь к корневой директории с данными
data_dir = 'data'
check_directory_structure(data_dir)