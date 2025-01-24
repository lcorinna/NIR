import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import numpy as np
import os
import shutil

# Параметры
img_width, img_height = 640, 640
batch_size = 32
num_classes = 3  # Количество классов дефектов

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

# Аугментация данных
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
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
    shuffle=False  # Важно для анализа результатов тестирования
)

# Загрузка уже обученной модели
if os.path.exists('aircraft_defect_classifier.keras'):
    model = load_model('aircraft_defect_classifier.keras')
    print("Модель загружена из файла.")
else:
    print("Файл модели не найден. Убедитесь, что вы уже провели обучение.")

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
with open('test_classification_report.txt', 'w') as f:
    f.write(report)

cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", cm)
