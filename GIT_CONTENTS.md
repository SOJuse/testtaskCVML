# Что в репозитории

## Включено:

### Основные файлы
- `README.md` - как пользоваться
- `report.md` - подробный отчет
- `data.yaml` - настройки датасета

### Модели
- `runs/detect/yolov11n_opt2/weights/best.pt` - лучшая YOLOv11n (~5.4 MB)
- `runs/detect/yolov11s_run/weights/best.pt` - лучшая YOLOv11s (~19 MB)

### Результаты и графики
- Графики обучения: results.png, train_batch0.jpg
- Метрики YOLOv11n_opt2: PR-curve, F1-curve, P-curve, R-curve, confusion matrix
- Метрики YOLOv11s: PR-curve, F1-curve, confusion matrix  
- Метрики YOLOv11n: PR-curve, F1-curve, confusion matrix
- Примеры батчей: val_batch0_pred.jpg
- Примеры предсказаний: по одному кадру от каждой модели

### Структура
- Папки dataset с .keep файлами (пустые, но структура видна)

## Не включено:
- Большие датасеты (изображения, разметка)
- Промежуточные веса (last.pt, epoch*.pt)
- Временные файлы, логи, виртуальные окружения

## Размер: ~35 MB