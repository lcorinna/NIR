import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import os
import shutil
import matplotlib.pyplot as plt

# Параметры
img_width, img_height = 640, 640  # Размер входного изображения
batch_size = 32  # Размер батча
epochs = 50  # Количество эпох
num_classes = 3  # Количество классов дефектов

# Пути к данным
data_dir = 'data'
train_data_dir = os.path.join(data_dir, 'train')
validation_data_dir = os.path.join(data_dir, 'valid')
test_data_dir = os.path.join(data_dir, 'test')

# Убедимся, что директории для изображений существуют
for directory in [train_data_dir, validation_data_dir, test_data_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Аугментация данных для повышения разнообразия тренировочных данных
train_datagen = ImageDataGenerator(
    rescale=1.0/255,  # Масштабирование значений пикселей в диапазон [0, 1]
    shear_range=0.2,  # Применение сдвига среза
    zoom_range=0.2,  # Применение масштабирования
    horizontal_flip=True  # Отражение изображений по горизонтали
)

# Аугментация для валидационных и тестовых данных (только нормализация)
validation_datagen = ImageDataGenerator(rescale=1.0/255)
test_datagen = ImageDataGenerator(rescale=1.0/255)

# Генераторы данных
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),  # Изменение размера изображений под модель
    batch_size=batch_size,
    class_mode='categorical'  # Категориальная классификация
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

# Создание модели
model = Sequential([
    Input(shape=(img_width, img_height, 3)),  # Входной слой с заданными размерами изображения
    Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.01)),  # Первый сверточный слой с L2-регуляризацией
    MaxPooling2D(pool_size=(2, 2)),  # Пуллинг для уменьшения размера карты признаков
    Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.01)),  # Второй сверточный слой
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.01)),  # Третий сверточный слой
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),  # Преобразование карты признаков в одномерный вектор
    Dense(256, activation='relu', kernel_regularizer=l2(0.01)),  # Полносвязный слой с L2-регуляризацией
    Dropout(0.5),  # Dropout для предотвращения переобучения
    Dense(num_classes, activation='softmax')  # Выходной слой с softmax для классификации
])

# Компиляция модели
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks для управления обучением
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),  # Ранняя остановка при отсутствии улучшений
    ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss', mode='min'),  # Сохранение лучшей модели
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)  # Уменьшение learning rate при плато
]

# Обучение модели
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    epochs=epochs,
    callbacks=callbacks
)

# Оценка модели на тестовых данных
test_loss, test_accuracy = model.evaluate(test_generator, steps=test_generator.samples // batch_size)
print(f'Test accuracy: {test_accuracy * 100:.2f}%')

# Предсказания и отчет
predictions = model.predict(test_generator, steps=len(test_generator))
y_pred = np.argmax(predictions, axis=-1)
y_true = test_generator.classes

# Анализ результатов
class_labels = list(test_generator.class_indices.keys())
report = classification_report(y_true, y_pred, target_names=class_labels, zero_division=1)
print(report)

# Сохранение отчета
with open('classification_report.txt', 'w') as f:
    f.write(report)

cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", cm)

# Сохранение структуры модели
model.save('final_model.keras')
with open('model_summary.txt', 'w', encoding='utf-8') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))

# Сохранение истории обучения
history_df = pd.DataFrame(history.history)
history_df.to_csv('training_history.csv', index=False)

# Построение графиков метрик
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('accuracy_plot.png')
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_plot.png')
plt.show()
