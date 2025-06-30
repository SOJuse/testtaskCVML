import os
from pathlib import Path
import cv2
import albumentations as A
import numpy as np

IMG_DIR = Path("dataset_balance/all_frames")
LABEL_DIR = Path("dataset_balance/labels/train")
AUGS_PER_IMAGE = 5

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.7),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.6),
    A.GaussNoise(p=0.4),
    A.MotionBlur(blur_limit=3, p=0.3),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

def read_labels(label_path):
    bboxes, class_labels = [], []
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id = int(parts[0])
                bbox = [float(x) for x in parts[1:]]
                bboxes.append(bbox)
                class_labels.append(class_id)
    return bboxes, class_labels

def save_labels(label_path, bboxes, class_labels):
    with open(label_path, 'w', encoding='utf-8') as f:
        for class_id, bbox in zip(class_labels, bboxes):
            f.write(f"{class_id} {' '.join(f'{x:.6f}' for x in bbox)}\n")

def main():
    for label_file in os.listdir(LABEL_DIR):
        if not label_file.endswith('.txt') or '_aug' in label_file:
            continue
        img_file = label_file.replace('.txt', '.jpg')
        img_path = IMG_DIR / img_file
        label_path = LABEL_DIR / label_file
        if not img_path.exists():
            continue
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        bboxes, class_labels = read_labels(label_path)
        if len(bboxes) == 0:
            continue
        for i in range(AUGS_PER_IMAGE):
            try:
                augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
                if len(augmented['bboxes']) == 0:
                    continue
                aug_img_name = f"{img_path.stem}_aug{i+1}.jpg"
                aug_label_name = f"{label_path.stem}_aug{i+1}.txt"
                cv2.imwrite(str(IMG_DIR / aug_img_name), augmented['image'])
                save_labels(LABEL_DIR / aug_label_name, augmented['bboxes'], augmented['class_labels'])
            except Exception as e:
                print(f"ошибка аугментации {img_file}: {e}")
                continue
    print("done")

if __name__ == '__main__':
    main() 