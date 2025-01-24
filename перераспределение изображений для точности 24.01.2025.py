import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50
from sklearn.model_selection import train_test_split
import numpy as np
import os
import shutil
import pandas as pd

# Параметры
img_width, img_height = 224, 224  # ResNet требует меньший размер изображений
batch_size = 32
num_classes = 3  # Количество классов дефектов

# Уникальные имена для сохранения результатов
model_filename = 'resnet50_defect_classifier.keras'
classification_report_filename = 'resnet50_classification_report.txt'
confusion_matrix_filename = 'resnet50_confusion_matrix.txt'

data_dir = 'data'
train_data_dir = os.path.join(data_dir, 'train')
validation_data_dir = os.path.join(data_dir, 'valid')
test_data_dir = os.path.join(data_dir, 'test')

# Убедимся, что директории существуют
for directory in [train_data_dir, validation_data_dir, test_data_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Функция для проверки структуры данных
def verify_data_split():
    for dataset_type, directory in zip(['Train', 'Validation', 'Test'], [train_data_dir, validation_data_dir, test_data_dir]):
        total_images = sum([len(files) for _, _, files in os.walk(directory)])
        print(f"{dataset_type} dataset: {total_images} images")

verify_data_split()

# Функция для перераспределения данных
def redistribute_images(data_dir, target_size_per_class=250):
    # Читаем CSV файлы
    train_classes = pd.read_csv(os.path.join(data_dir, 'train_classes.csv'))
    valid_classes = pd.read_csv(os.path.join(data_dir, 'valid_classes.csv'))
    test_classes = pd.read_csv(os.path.join(data_dir, 'test_classes.csv'))

    # Удаление пробелов из названий столбцов
    train_classes.columns = train_classes.columns.str.strip()
    valid_classes.columns = valid_classes.columns.str.strip()
    test_classes.columns = test_classes.columns.str.strip()

    # Словарь для хранения файлов по классам
    class_files = {"Dent": [], "Fastener Damage": [], "Rupture": []}

    # Собираем данные из train и valid
    for dataset, dataset_name in zip([train_classes, valid_classes], ["train", "valid"]):
        for index, row in dataset.iterrows():
            filename = row['filename']
            class_label = row[['Dent', 'Fastener Damage', 'Rupture']].idxmax()
            full_path = os.path.join(data_dir, dataset_name, class_label, filename)
            if os.path.exists(full_path):
                class_files[class_label].append(full_path)

    # Создаём сбалансированный тестовый набор
    for class_name, files in class_files.items():
        test_class_dir = os.path.join(test_data_dir, class_name)
        if not os.path.exists(test_class_dir):
            os.makedirs(test_class_dir)

        # Убираем лишние файлы из текущего тестового набора
        existing_files = os.listdir(test_class_dir)
        for file in existing_files:
            os.remove(os.path.join(test_class_dir, file))

        # Берём первые target_size_per_class файлов для каждого класса
        selected_files = files[:target_size_per_class]

        for file_path in selected_files:
            dest_path = os.path.join(test_class_dir, os.path.basename(file_path))
            shutil.copy(file_path, dest_path)

    print("Перераспределение завершено. Тестовый набор сбалансирован.")

redistribute_images(data_dir, target_size_per_class=250)

# Аугментация данных с усилением для уменьшения дисбаланса
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2
)
validation_datagen = ImageDataGenerator(rescale=1.0/255)
test_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Использование предобученной модели ResNet50
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# Замораживаем веса базовой модели
base_model.trainable = False

# Создаём собственный классификатор поверх ResNet50
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Обучение модели
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    epochs=1  # Уменьшили количество эпох для теста
)

# Сохранение модели
model.save(model_filename)
print(f"Model saved as {model_filename}")

# Оценка модели на тестовых данных
test_loss, test_accuracy = model.evaluate(test_generator, steps=test_generator.samples // batch_size)
print(f'Test accuracy: {test_accuracy * 100:.2f}%')

# Предсказания и отчет
predictions = model.predict(test_generator, steps=len(test_generator))
y_pred = np.argmax(predictions, axis=-1)
y_true = test_generator.classes

# Анализ результатов
from sklearn.metrics import classification_report, confusion_matrix
class_labels = list(test_generator.class_indices.keys())
report = classification_report(y_true, y_pred, target_names=class_labels, zero_division=1)
print(report)

# Сохранение отчета
with open(classification_report_filename, 'w') as f:
    f.write(report)
print(f"Classification report saved as {classification_report_filename}")

# Сохранение матрицы ошибок
cm = confusion_matrix(y_true, y_pred)
with open(confusion_matrix_filename, 'w') as f:
    f.write("Confusion Matrix:\n")
    np.savetxt(f, cm, fmt='%d')
print(f"Confusion matrix saved as {confusion_matrix_filename}")
