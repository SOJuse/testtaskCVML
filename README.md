# PredictYourFood

## Доступные модели

### YOLOv11n (nano, оптимизированная)
- **Размер:** ~2.6 млн параметров
- **Скорость инференса:** ~67-91 мс/изображение (на CPU)
- **mAP50-95:** до 0.95 (после оптимизации)
- **Веса:** `runs/detect/yolov11n_opt2/weights/best.pt`
- **Лучше всего подходит:** для быстрого инференса и ограниченных ресурсов

### YOLOv11s (small)
- **Размер:** ~9.4 млн параметров
- **Скорость инференса:** ~183 мс/изображение (на CPU)
- **mAP50-95:** до 0.957
- **Веса:** `runs/detect/yolov11s/weights/best.pt`
- **Лучше всего подходит:** для максимального качества, если скорость не критична

## Быстрый старт

1. **Выбор модели:**
   - Для быстрого старта и теста — YOLOv11n (nano, оптимизированная)
   - Для максимального качества — YOLOv11s

2. **Обучение (пример для YOLOv11n, оптимизированная версия):**

```bash
yolo train model=yolo11n.pt data=data.yaml epochs=70 imgsz=640 name=yolov11n_opt2 lr0=0.004 batch=16 optimizer=Adam lrf=0.05 scale=0.5 translate=0.12 hsv_h=0.018 hsv_s=0.7
```

3. **Валидация на тестовой выборке:**

```bash
yolo val model=runs/detect/yolov11n_opt2/weights/best.pt data=data.yaml split=test
```

4. **Предикт на новых изображениях:**

```bash
yolo predict model=runs/detect/yolov11n_opt2/weights/best.pt source=dataset/images/test
```

5. **Обучение YOLOv11s:**

```bash
yolo train model=yolo11s.pt data=data.yaml epochs=50 imgsz=640 name=yolov11s
```

## Структура проекта
- `dataset/images/{train,val,test}` — изображения
- `dataset/labels/{train,val,test}` — разметка в формате YOLO
- `runs/detect/` — результаты обучения и веса моделей
- `data.yaml` — описание датасета для YOLO

## Автор
Ларкин Григорий