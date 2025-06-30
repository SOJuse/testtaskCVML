# Модели и их использование

## Доступные модели

- **train5** — самая свежая модель (оптимизированный датасет, много эпох)
  - Веса: `runs/detect/train5/weights/best.pt`, `last.pt`, а также веса по эпохам (`epoch10.pt`, ...)
- **train3** — предыдущая версия (другой конфиг/датасет)
  - Веса: `runs/detect/train3/weights/best.pt`, `last.pt`, веса по эпохам

## Как использовать модели

### 1. Инференс (детекция на изображениях/видео)

```bash
yolo predict model=runs/detect/train5/weights/best.pt source=path/to/image_or_video
```
- Можно заменить путь к модели на любую из доступных (например, train3)
- Для видео и папок с изображениями — просто указывай нужный source

### 2. Дообучение (fine-tune)

```bash
yolo train model=runs/detect/train5/weights/best.pt data=path/to/data.yaml cfg=path/to/config.yaml
```
- Можно дообучать с любого веса (best.pt, last.pt, epoch*.pt)
- Указывай свой конфиг и датасет

### 3. Графики и метрики

- Все графики (loss, precision, recall, confusion matrix и др.) лежат в `runs/detect/train*/` и `runs/detect/val*/`
- Открывай PNG/JPG для анализа качества

---