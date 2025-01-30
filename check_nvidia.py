import os
import glob

# Функция для чтения аннотаций YOLO сегментации и преобразования их в формат детекции
def convert_segmentation_to_detection(input_dir, output_dir, image_width, image_height):
    # Убедимся, что выходная папка существует
    os.makedirs(output_dir, exist_ok=True)

    # Получаем все файлы аннотаций
    annotation_files = glob.glob(os.path.join(input_dir, "*.txt"))

    for annotation_file in annotation_files:
        with open(annotation_file, "r") as file:
            lines = file.readlines()

        # Проверка: если файл уже в формате детекции, пропускаем обработку
        is_detection_format = all(len(line.strip().split()) == 5 for line in lines)
        if is_detection_format:
            # Копируем файл в выходную папку без изменений
            output_file = os.path.join(output_dir, os.path.basename(annotation_file))
            with open(output_file, "w") as file:
                file.write("".join(lines))
            continue

        converted_lines = []

        for line in lines:
            data = line.strip().split()

            # Первый элемент — class_id, остальные — координаты сегментации (x1, y1, x2, y2, ...)
            class_id = data[0]
            points = list(map(float, data[1:]))

            # Разделяем x и y координаты
            x_coords = points[0::2]
            y_coords = points[1::2]

            # Проверяем и корректно рассчитываем bounding box
            x_min = min(x_coords)
            y_min = min(y_coords)
            x_max = max(x_coords)
            y_max = max(y_coords)

            # Вычисляем центр и размеры
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            width = x_max - x_min
            height = y_max - y_min

            # Если исходные координаты нормализованы, не выполняем дополнительное деление
            if image_width > 1 or image_height > 1:
                x_center /= image_width
                y_center /= image_height
                width /= image_width
                height /= image_height

            # Формируем строку для формата YOLO детекции
            converted_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            converted_lines.append(converted_line)

        # Записываем результат в выходной файл
        output_file = os.path.join(output_dir, os.path.basename(annotation_file))
        with open(output_file, "w") as file:
            file.write("\n".join(converted_lines))

# Параметры датасета
input_dir = r"E:/MAI/NIR/Aircraft_Defect_Detection/labels/"  # Папка с аннотациями сегментации
output_dir = r"E:/MAI/NIR/Aircraft_Defect_Detection/labels_new/"   # Папка для сохранения аннотаций детекции
image_width = 1  # Используем 1, если координаты уже нормализованы
image_height = 1 # Используем 1, если координаты уже нормализованы

# Запускаем конвертацию
convert_segmentation_to_detection(input_dir, output_dir, image_width, image_height)

print("Конвертация завершена!")
