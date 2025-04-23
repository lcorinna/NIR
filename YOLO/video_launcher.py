import cv2
import os
from ultralytics import YOLO
import torch

# === Проверка CUDA ===
print("[INFO] CUDA доступна:", torch.cuda.is_available())
print("[INFO] Используемое устройство:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

# === Параметры ===
VIDEO_PATH = "E:/MAI/NIR/test_data/video/7.mp4"  # ← Путь до входного видео
MODEL_PATH = "E:/MAI/NIR/YOLO/runs/diploma_training/experiment_name2/weights/best.pt"  # ← Путь до модели

# === Автоматическое формирование OUTPUT_PATH ===
video_filename = os.path.splitext(os.path.basename(VIDEO_PATH))[0]  # gorizontal
video_dir = os.path.dirname(VIDEO_PATH)
OUTPUT_PATH = os.path.join(video_dir, f"{video_filename}_output.mp4")

# === Загрузка модели YOLO ===
print("[INFO] Загрузка модели YOLO...")
model = YOLO(MODEL_PATH)

# === Открытие видео ===
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"[ERROR] Не удалось открыть видео: {VIDEO_PATH}")
    exit()

# === Получение параметров видео ===
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"[INFO] Видео: {frame_width}x{frame_height} @ {fps} FPS, всего кадров: {total_frames}")

# === Создание выходного видеофайла ===
out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# === Обработка видео покадрово ===
frame_idx = 0
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    results = model(frame)  # YOLO обрабатывает кадр
    annotated_frame = results[0].plot()  # Отрисовка боксов

    out.write(annotated_frame)  # Запись кадра в выходное видео

    # Отображение результата (опционально)
    # cv2.imshow("YOLO Detection", annotated_frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

    frame_idx += 1
    if frame_idx % 10 == 0:
        print(f"[INFO] Обработано кадров: {frame_idx}/{total_frames}")

# === Завершение ===
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"[INFO] Готово! Сохранено в {OUTPUT_PATH}")
