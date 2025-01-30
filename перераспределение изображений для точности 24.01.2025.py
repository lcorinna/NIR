import time
from ultralytics import YOLO

def train_model(device):
    print(f"Начало обучения на устройстве: {device}")
    start_time = time.time()

    # Загрузка модели
    model = YOLO("yolo11n.pt")

    # Обучение модели
    model.train(
        data="Aircraft_Defect_Detection/data.yaml",
        epochs=1,           # Одна эпоха для тестирования
        imgsz=640,          # Размер изображений
        batch=32,           # Размер батча
        device=device,      # Устройство для обучения (cuda или cpu)
        amp=True            # Смешанная точность для GPU
    )

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Обучение завершено на {device}. Затраченное время: {elapsed_time:.2f} секунд")
    return elapsed_time

if __name__ == '__main__':
    # Тестирование на CPU
    cpu_time = train_model('cpu')

    # Тестирование на CUDA, если доступно
    if torch.cuda.is_available():
        cuda_time = train_model('cuda')
    else:
        cuda_time = None

    # Вывод результатов
    print("\n=== Результаты ===")
    print(f"CPU время: {cpu_time:.2f} секунд")
    if cuda_time is not None:
        print(f"CUDA время: {cuda_time:.2f} секунд")
        if cuda_time < cpu_time:
            print("GPU (CUDA) быстрее для обучения.")
        else:
            print("CPU быстрее для обучения.")
    else:
        print("CUDA недоступна.")
