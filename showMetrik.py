from tensorflow.keras.models import load_model
import numpy as np

# Загрузка модели из файла .h5
model = load_model('./aircraft_defect_classifier.h5')

# Если у вас есть тестовые данные (X_test и y_test), выполните оценку:
loss, accuracy = model.evaluate(X_test, y_test)

print(f'Loss: {loss}')
print(f'Accuracy: {accuracy}')
