import os
import shutil
from random import shuffle


def split_dataset(base_path, train_percent, validate_percent):
    assert 0 < train_percent <= 100, "Train percent must be between 0 and 100"
    assert 0 < validate_percent <= 100, "Validate percent must be between 0 and 100"
    assert train_percent + validate_percent <= 100, "Sum of train and validate percent must be less than or equal to 100"

    # Определяем, нужно ли создавать тестовую часть
    create_test = (train_percent + validate_percent < 100)
    test_percent = 100 - train_percent - validate_percent if create_test else 0

    # Создаем директории для train и validate (и test, если необходимо)
    for folder in ['train', 'validate']:
        os.makedirs(os.path.join(base_path, folder, 'images'), exist_ok=True)
        os.makedirs(os.path.join(base_path, folder, 'labels'), exist_ok=True)

    if create_test:
        os.makedirs(os.path.join(base_path, 'test', 'images'), exist_ok=True)
        os.makedirs(os.path.join(base_path, 'test', 'labels'), exist_ok=True)

    # Получаем списки файлов изображений и меток
    images = sorted(os.listdir(os.path.join(base_path, 'images')))
    labels = sorted(os.listdir(os.path.join(base_path, 'labels')))

    # Перемешиваем изображения для случайного разделения
    shuffle(images)

    # Вычисляем индексы для разделения наборов данных
    n_total = len(images)
    n_train = int(n_total * train_percent / 100)
    n_validate = int(n_total * validate_percent / 100)
    n_test = n_total - n_train - n_validate

    # Функция для перемещения файлов
    def move_files(start_idx, end_idx, destination):
        for i in range(start_idx, end_idx):
            image_name = images[i]
            label_name = os.path.splitext(image_name)[0] + '.txt'
            shutil.move(
                os.path.join(base_path, 'images', image_name),
                os.path.join(base_path, destination, 'images', image_name)
            )
            if label_name in labels:
                shutil.move(
                    os.path.join(base_path, 'labels', label_name),
                    os.path.join(base_path, destination, 'labels', label_name)
                )

    # Перемещаем файлы в соответствующие папки
    move_files(0, n_train, 'train')
    move_files(n_train, n_train + n_validate, 'validate')

    if create_test:
        move_files(n_train + n_validate, n_total, 'test')


# Пример использования скрипта
split_dataset(r"E:/MAI/NIR/Aircraft_Defect_Detection/", 70, 20)
