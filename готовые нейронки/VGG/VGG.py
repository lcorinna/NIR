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
img_width, img_height = 224, 224
batch_size = 32
epochs = 50  # Увеличить количество эпох до 50
num_classes = 3  # Количество классов дефектов

# Пути к данным
data_dir = '../../data'
train_data_dir = os.path.join(data_dir, 'train')
validation_data_dir = os.path.join(data_dir, 'valid')
test_data_dir = os.path.join(data_dir, 'test')

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

# Создание модели VGG-16
model = Sequential([
    Input(shape=(img_width, img_height, 3)),
    
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    MaxPooling2D(pool_size=(2, 2), strides=2),

    Conv2D(128, (3, 3), padding='same', activation='relu'),
    Conv2D(128, (3, 3), padding='same', activation='relu'),
    MaxPooling2D(pool_size=(2, 2), strides=2),

    Conv2D(256, (3, 3), padding='same', activation='relu'),
    Conv2D(256, (3, 3), padding='same', activation='relu'),
    Conv2D(256, (3, 3), padding='same', activation='relu'),
    MaxPooling2D(pool_size=(2, 2), strides=2),

    Conv2D(512, (3, 3), padding='same', activation='relu'),
    Conv2D(512, (3, 3), padding='same', activation='relu'),
    Conv2D(512, (3, 3), padding='same', activation='relu'),
    MaxPooling2D(pool_size=(2, 2), strides=2),

    Conv2D(512, (3, 3), padding='same', activation='relu'),
    Conv2D(512, (3, 3), padding='same', activation='relu'),
    Conv2D(512, (3, 3), padding='same', activation='relu'),
    MaxPooling2D(pool_size=(2, 2), strides=2),

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

pd.DataFrame([hyperparameters]).to_csv('vgg16_hyperparameters.csv', index=False)

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
    with open('vgg16_classification_report.txt', 'w') as f:
        f.write(report)

    # Вывод матрицы ошибок
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    # Сохранение модели
    model.save('vgg16_aircraft_defect_classifier.keras')

    # Сохранение структуры модели в текстовый файл
    with open('vgg16_model_summary.txt', 'w', encoding='utf-8') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    # --- Сохранение метрик обучения ---
    history_df = pd.DataFrame(history.history)
    history_df.to_csv('vgg16_training_history.csv', index=False)

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
plt.savefig('vgg16_accuracy_plot.png')
plt.show()

# Построение графика потерь
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('vgg16_loss_plot.png')
plt.show()
