import os
import shutil
import random
from pathlib import Path
import cv2
import albumentations as A
import numpy as np

TRAIN_RATIO, VAL_RATIO = 0.7, 0.15
AUGS_PER_IMAGE = 6

CVAT_ANNOTATIONS = "job_2620809_annotations_2025_06_28_19_08_41_yolo 1.1/obj_train_data"
IMAGES_PATH = "dataset_balance/images/all_frames"
OUTPUT_PATH = "dataset_balance"

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.7),
    A.ShiftScaleRotate(
        shift_limit=0.1,
        scale_limit=0.1,
        rotate_limit=10,
        p=0.5
    ),
    
    A.RandomBrightnessContrast(
        brightness_limit=0.3,
        contrast_limit=0.3,
        p=0.8
    ),
    
    A.HueSaturationValue(
        hue_shift_limit=20,
        sat_shift_limit=30,
        val_shift_limit=20,
        p=0.6
    ),
    
    A.GaussNoise(p=0.4),
    A.MotionBlur(blur_limit=3, p=0.3),
    
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

def get_all_pairs():
    pairs = []
    annotation_files = os.listdir(CVAT_ANNOTATIONS)
    
    for annotation_file in annotation_files:
        if annotation_file.endswith('.txt'):
            image_file = annotation_file.replace('.txt', '.jpg')
            image_path = os.path.join(IMAGES_PATH, image_file)
            annotation_path = os.path.join(CVAT_ANNOTATIONS, annotation_file)
            
            if os.path.exists(image_path):
                pairs.append((image_path, annotation_path))
    
    return pairs

def read_labels(label_path):
    bboxes, class_labels = [], []
    try:
        with open(label_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    bbox = [float(x) for x in parts[1:]]
                    bboxes.append(bbox)
                    class_labels.append(class_id)
    except Exception as e:
        print("ошибка чтения")
    
    return bboxes, class_labels

def save_labels(label_path, bboxes, class_labels):
    try:
        with open(label_path, 'w', encoding='utf-8') as f:
            for class_id, bbox in zip(class_labels, bboxes):
                f.write(f"{class_id} {' '.join(f'{x:.6f}' for x in bbox)}\n")
    except Exception as e:
        print("ошибка сохранения")


def main():
    output_path = Path(OUTPUT_PATH)
    for split in ['train', 'val', 'test']:
        (output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_path / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    pairs = get_all_pairs()
    if len(pairs) == 0:
        return
    
    random.seed(42)
    random.shuffle(pairs)
    
    n_total = len(pairs)
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)
    
    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:n_train + n_val]
    test_pairs = pairs[n_train + n_val:]
    
    train_count = 0
    for img_path, label_path in train_pairs:
        img_name = Path(img_path).name
        label_name = Path(label_path).name
        
        if not os.path.exists(img_path):
            continue
        if not os.path.exists(label_path):
            continue
        
        shutil.copy2(img_path, output_path / 'images' / 'train' / img_name)
        shutil.copy2(label_path, output_path / 'labels' / 'train' / label_name)
        train_count += 1
        
        image = cv2.imread(img_path)
        if image is None:
            continue
        
        bboxes, class_labels = read_labels(label_path)
        if len(bboxes) == 0:
            continue
        
        for i in range(AUGS_PER_IMAGE):
            try:
                augmented = transform(
                    image=image,
                    bboxes=bboxes,
                    class_labels=class_labels
                )
                if len(augmented['bboxes']) == 0:
                    continue
                aug_img_name = f"{Path(img_path).stem}_aug{i+1}.jpg"
                aug_label_name = f"{Path(label_path).stem}_aug{i+1}.txt"
                cv2.imwrite(
                    str(output_path / 'images' / 'train' / aug_img_name),
                    augmented['image']
                )
                save_labels(
                    output_path / 'labels' / 'train' / aug_label_name,
                    augmented['bboxes'],
                    augmented['class_labels']
                )
                train_count += 1
            except Exception as e:
                print("ошибка аугментации")
                continue
    
    for img_path, label_path in val_pairs:
        img_name = Path(img_path).name
        label_name = Path(label_path).name
        if not os.path.exists(img_path):
            continue
        if not os.path.exists(label_path):
            continue
        shutil.copy2(img_path, output_path / 'images' / 'val' / img_name)
        shutil.copy2(label_path, output_path / 'labels' / 'val' / label_name)
    
    for img_path, label_path in test_pairs:
        img_name = Path(img_path).name
        label_name = Path(label_path).name
        if not os.path.exists(img_path):
            continue
        if not os.path.exists(label_path):
            continue
        shutil.copy2(img_path, output_path / 'images' / 'test' / img_name)
        shutil.copy2(label_path, output_path / 'labels' / 'test' / label_name)
    
    # Статистика
    train_images = len(list((output_path / 'images' / 'train').glob('*.jpg')))
    val_images = len(list((output_path / 'images' / 'val').glob('*.jpg')))
    test_images = len(list((output_path / 'images' / 'test').glob('*.jpg')))
    
    print("done")

if __name__ == '__main__':
    main() 