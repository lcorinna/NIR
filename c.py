import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

# Параметры
img_width, img_height = 640, 640

# Путь к сохраненной модели
model_path = 'aircraft_defect_classifier.h5'

# Загрузка модели
model = tf.keras.models.load_model(model_path)

# Предобработка изображения
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(img_width, img_height))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Масштабирование
    return img_array

# Предсказание класса
def predict_image_class(image_path):
    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array)
    class_indices = {0: 'Dent', 1: 'Fastener Damage', 2: 'Rupture'}
    predicted_class = class_indices[np.argmax(predictions)]
    return predicted_class

# Пример использования
new_image_path = 'cat.jpg'
predicted_class = predict_image_class(new_image_path)
print(f'The predicted class for the image is: {predicted_class}')
