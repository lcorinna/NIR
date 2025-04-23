import os
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix

# Параметры
img_width, img_height = 640, 640
batch_size = 32
epochs = 50
num_classes = 3

# Пути
data_dir = 'data'
train_data_dir = os.path.join(data_dir, 'train')
validation_data_dir = os.path.join(data_dir, 'valid')
test_data_dir = os.path.join(data_dir, 'test')

# Аугментация
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_val_datagen = ImageDataGenerator(rescale=1.0/255)

# Генераторы
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = test_val_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_val_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False  # Важно для меток
)

# Модель
model = Sequential([
    Input(shape=(img_width, img_height, 3)),
    Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.01)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.01)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.01)),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Компиляция
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss', mode='min'),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
]

# Обучение
history = model.fit(
    train_generator,
    steps_per_epoch=math.ceil(train_generator.samples / batch_size),
    validation_data=validation_generator,
    validation_steps=math.ceil(validation_generator.samples / batch_size),
    epochs=epochs,
    callbacks=callbacks
)

# Оценка
test_loss, test_accuracy = model.evaluate(test_generator, steps=math.ceil(test_generator.samples / batch_size))
print(f'Test accuracy: {test_accuracy * 100:.2f}%')

# Предсказания
predictions = model.predict(test_generator, steps=math.ceil(test_generator.samples / batch_size))
y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes

# Классы
class_labels = list(test_generator.class_indices.keys())

# Отчёт
report = classification_report(y_true, y_pred, target_names=class_labels, zero_division=1)
print(report)
with open('classification_report.txt', 'w') as f:
    f.write(report)

# Матрица ошибок
cm = confusion_matrix(y_true, y_pred)
np.savetxt("confusion_matrix.csv", cm, fmt='%d', delimiter=",")

# Сохраняем модель
model.save('final_model.keras')
with open('model_summary.txt', 'w', encoding='utf-8') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))

# Сохраняем историю обучения
history_df = pd.DataFrame(history.history)
history_df.to_csv('training_history.csv', index=False)

# Графики
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('accuracy_plot.png')
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('loss_plot.png')
plt.close()

# Сохраняем гиперпараметры
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
with open('hyperparameters.json', 'w') as f:
    json.dump(hyperparameters, f, indent=4)
