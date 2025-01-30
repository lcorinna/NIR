import cv2
import os

# Уникальные цвета для каждого ключа в label_map
class_colors = {
    0: (255, 0, 0),  # Красный
    1: (0, 255, 0),  # Зеленый
    2: (0, 0, 255),  # Синий
    3: (255, 255, 0),  # Циан
    4: (255, 0, 255),  # Маджента
    5: (0, 255, 255),  # Желтый
    6: (128, 128, 0),  # Оливковый
    7: (128, 0, 0),  # Темно-красный
    8: (128, 128, 128),  # Серый
    9: (0, 128, 0)  # Темно-зеленый
}

def draw_boxes(images_path, labels_path, output_path, label_map, class_colors, draw_height=False):
    image_files = [f for f in os.listdir(images_path) if f.endswith(".jpg") or f.endswith(".png")]
    total_images = len(image_files)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for index, filename in enumerate(image_files):
        image_file_path = os.path.join(images_path, filename)
        label_file_path = os.path.join(labels_path, filename.replace(".jpg", ".txt").replace(".png", ".txt"))
        output_file_path = os.path.join(output_path, filename)

        image = cv2.imread(image_file_path)
        height, width, _ = image.shape

        valid_lines = []
        if os.path.exists(label_file_path):
            with open(label_file_path, 'r') as file:
                for line in file:
                    if line.strip():
                        values = line.split()
                        if len(values) != 5:
                            print(f"Invalid line in {label_file_path}: {line.strip()}")
                            continue

                        class_id, x_center, y_center, w, h = map(float, values)

                        # Используем class_id как ключ для цвета
                        color = class_colors.get(int(class_id), (255, 255, 255))  # Белый по умолчанию

                        x_center, y_center, w, h = x_center * width, y_center * height, w * width, h * height
                        x_min, y_min = int(x_center - w / 2), int(y_center - h / 2)

                        class_name = label_map.get(int(class_id), "Unknown")
                        cv2.rectangle(image, (x_min, y_min), (int(x_min + w), int(y_min + h)), color, 2)

                        # Добавляем высоту, если draw_height=True
                        if draw_height:
                            height_text = f"H: {int(h)}px"
                            cv2.putText(image, height_text, (x_min, y_min + int(h) + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                        cv2.putText(image, class_name, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                        valid_lines.append(line.strip())

            with open(label_file_path, 'w') as file:
                for valid_line in valid_lines:
                    file.write(valid_line + "\n")

        cv2.imwrite(output_file_path, image)
        print(f"Processing {index + 1} / {total_images}")

    print("Processing completed.")

# Словарь классов
label_map = {0: 'person', 1: 'e-Sim', 2: 'Bicycle', 3:'Person_on_eSim', 4:'Person_on_Bicycle', 5:'Two_Person_on_eSim', 6:'Scooter', 7:'Person_on_Scooter'}
# label_map = {0: 'e-Sim', 1:'Person_on_eSim', 2:'Two_Person_on_eSim'}
label_map = {0: 'Dent', 1:'Fastener Damage', 2: 'Rupture'}

image_path = r"E:/MAI/NIR/Aircraft_Defect_Detection/images/"
label_path = r"E:/MAI/NIR/Aircraft_Defect_Detection/labels_new/"
output_path = r"E:/MAI/NIR/Aircraft_Defect_Detection/images_with_boxes"

draw_boxes(image_path, label_path, output_path, label_map, class_colors, draw_height=True)
