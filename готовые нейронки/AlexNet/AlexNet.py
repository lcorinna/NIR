import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import os
import matplotlib.pyplot as plt

# Параметры
img_width, img_height = 640, 640
batch_size = 32
epochs = 50  # Увеличить количество эпох до 50
num_classes = 3  # Количество классов дефектов

# Пути к данным
data_dir = '../../data'
train_data_dir = os.path.join(data_dir, 'train')
validation_data_dir = os.path.join(data_dir, 'valid')
test_data_dir = os.path.join(data_dir, 'test')

# Функция для чтения меток классов из поддиректории
def load_class_labels(directory):
    class_labels_file = os.path.join(directory, '_classes.csv')
    return pd.read_csv(class_labels_file)

# Убедимся, что директории для изображений существуют
for directory in [train_data_dir, validation_data_dir, test_data_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Чтение и обработка меток классов
train_labels_df = load_class_labels(train_data_dir)
val_labels_df = load_class_labels(validation_data_dir)
test_labels_df = load_class_labels(test_data_dir)

# Удаление ведущих пробелов в названиях колонок
for df in [train_labels_df, val_labels_df, test_labels_df]:
    df.columns = df.columns.str.strip()

# Создание новых колонок с комбинированными метками классов
for df in [train_labels_df, val_labels_df, test_labels_df]:
    df['label'] = df[['Dent', 'Fastener Damage', 'Rupture']].idxmax(axis=1)

# Проверка наличия изображений в исходной директории
src_images_dir = os.path.join(data_dir, 'images')
if not os.path.exists(src_images_dir):
    print(f"Source images directory {src_images_dir} does not exist!")
else:
    print(f"Source images directory {src_images_dir} found.")

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
    class_mode='categorical'
)

# Создание модели AlexNet
model = Sequential([
    Input(shape=(img_width, img_height, 3)),
    
    Conv2D(96, (11, 11), strides=4, activation='relu'),
    MaxPooling2D(pool_size=(3, 3), strides=2),
    
    Conv2D(256, (5, 5), padding='same', activation='relu'),
    MaxPooling2D(pool_size=(3, 3), strides=2),
    
    Conv2D(384, (3, 3), padding='same', activation='relu'),
    Conv2D(384, (3, 3), padding='same', activation='relu'),
    Conv2D(256, (3, 3), padding='same', activation='relu'),
    MaxPooling2D(pool_size=(3, 3), strides=2),
    
    Flatten(),
    Dense(4096, activation='relu'),
    Dropout(0.5),
    Dense(4096, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Сохранение гиперпараметров
hyperparameters = {
    'img_width': img_width,
    'img_height': img_height,
    'batch_size': batch_size,
    'epochs': epochs,
    'optimizer': 'Adam',
    'learning_rate': 0.001,
    'loss_function': 'categorical_crossentropy',
    'num_classes': num_classes
}

pd.DataFrame([hyperparameters]).to_csv('alexnet_hyperparameters.csv', index=False)

# Обучение модели
if train_generator.samples == 0 or validation_generator.samples == 0:
    print("No images found for training or validation. Please check your dataset.")
else:
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size,
        epochs=epochs
    )

    # Оценка модели на тестовых данных
    test_loss, test_accuracy = model.evaluate(test_generator, steps=test_generator.samples // batch_size)
    print(f'Test accuracy: {test_accuracy * 100:.2f}%')

    # Получение предсказаний на тестовых данных
    predictions = model.predict(test_generator, steps=len(test_generator))
    y_pred = np.argmax(predictions, axis=-1)
    y_true = test_generator.classes

    # Вывод отчета по классификации
    class_labels = list(test_generator.class_indices.keys())
    report = classification_report(y_true, y_pred, target_names=class_labels, zero_division=1)
    print(report)

    # Сохранение отчета в файл
    with open('alexnet_classification_report.txt', 'w') as f:
        f.write(report)

    # Вывод матрицы ошибок
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    # Сохранение модели
    model.save('alexnet_aircraft_defect_classifier.keras')

    # Сохранение структуры модели в текстовый файл
    with open('alexnet_model_summary.txt', 'w', encoding='utf-8') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    # --- Сохранение метрик обучения ---
    history_df = pd.DataFrame(history.history)
    history_df.to_csv('alexnet_training_history.csv', index=False)

    # Вывод финальной точности и потерь на тренировочных и валидационных данных
    training_accuracy = history.history['accuracy'][-1]
    validation_accuracy = history.history['val_accuracy'][-1]
    training_loss = history.history['loss'][-1]
    validation_loss = history.history['val_loss'][-1]

    print(f'Final training accuracy: {training_accuracy * 100:.2f}%')
    print(f'Final validation accuracy: {validation_accuracy * 100:.2f}%')
    print(f'Final training loss: {training_loss:.4f}')
    print(f'Final validation loss: {validation_loss:.4f}')

# Построение графиков метрик
# Построение графика точности
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('alexnet_accuracy_plot.png')
plt.show()

# Построение графика потерь
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('alexnet_loss_plot.png')
plt.show()
