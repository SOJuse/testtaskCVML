import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

SRC_IMG_DIR = Path("dataset/images/all_frames")
SRC_LABEL_DIR = Path("job_2620809_annotations_2025_06_28_19_08_41_yolo 1.1/obj_train_data")
DST_IMG_DIR = Path("dataset_balance/all_frames")
DST_LABEL_DIR = Path("dataset_balance/labels/train")

DST_IMG_DIR.mkdir(parents=True, exist_ok=True)
DST_LABEL_DIR.mkdir(parents=True, exist_ok=True)

label_files = [f for f in SRC_LABEL_DIR.iterdir() if f.suffix == ".txt"]
class_counts = defaultdict(int)
file_classes = {}

for file in label_files:
    with file.open() as f:
        lines = f.readlines()
    classes = [int(float(line.split()[0])) for line in lines if line.strip()]
    base = file.stem
    file_classes[base] = classes
    for cls in classes:
        class_counts[cls] += 1

min_count = min(class_counts.values())
target_count = int(min_count * 7)

chosen = set()
used = defaultdict(int)
items = list(file_classes.items())
random.seed(42)
random.shuffle(items)

for base, classes in items:
    if not any(used[c] < target_count for c in set(classes)):
        continue
    temp = used.copy()
    skip = False
    for c in classes:
        temp[c] += 1
        if temp[c] > target_count:
            skip = True
            break
    if skip:
        continue
    chosen.add(base)
    for c in classes:
        used[c] += 1

for base in chosen:
    img_path = SRC_IMG_DIR / f"{base}.jpg"
    label_path = SRC_LABEL_DIR / f"{base}.txt"
    if img_path.exists() and label_path.exists():
        shutil.copy2(img_path, DST_IMG_DIR / img_path.name)
        shutil.copy2(label_path, DST_LABEL_DIR / label_path.name)
    else:
        print(f"Пропущено: {img_path} или {label_path} не найдено")

yaml_src = Path("dataset/data.yaml")
yaml_dst = Path("dataset_balance/data.yaml")
if yaml_src.exists():
    shutil.copy2(yaml_src, yaml_dst)