# from ultralytics import YOLO

# # Загрузка модели
# model = YOLO(r'E:\MAI\NIR\runs\diploma_training\experiment_12\weights\best.pt')  # Укажите путь к вашей модели

# # Выполнение инференса
# results = model('E:/MAI/NIR/изображения для проверки/image6.jpg', save=True, save_txt=True)

# # Отображение результатов
# for result in results:  # results — это список объектов
#     result.show()  # Показываем результат для каждого изображения

print(torch.cuda.is_available())