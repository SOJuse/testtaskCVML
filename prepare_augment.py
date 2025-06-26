import os
import shutil
import random
from pathlib import Path
import cv2
import albumentations as A

FOLDERS = ['frames/1', 'frames/2_1', 'frames/3_1', 'frames/3_2', 'frames/4']
TRAIN_RATIO, VAL_RATIO = 0.7, 0.15
AUGS_PER_IMAGE = 5

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.Rotate(limit=20, p=0.5),
    A.GaussNoise(p=0.3),
    A.MotionBlur(p=0.2),
    A.ColorJitter(p=0.3),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

def all_pairs():
    pairs = []
    for folder in FOLDERS:
        for f in os.listdir(folder):
            if f.endswith('.jpg'):
                img = os.path.join(folder, f)
                lbl = os.path.splitext(img)[0] + '.txt'
                if os.path.exists(lbl):
                    pairs.append((img, lbl))
    return pairs

def read_labels(lbl):
    b, c = [], []
    with open(lbl) as f:
        for l in f:
            p = l.strip().split()
            if len(p) == 5:
                c.append(int(p[0]))
                b.append([float(x) for x in p[1:]])
    return b, c

def save_labels(lbl, b, c):
    with open(lbl, 'w') as f:
        for cl, bb in zip(c, b):
            f.write(f"{cl} {' '.join(f'{x:.6f}' for x in bb)}\n")

if __name__ == '__main__':
    random.seed(42)
    out = Path('dataset')
    for s in ['train', 'val', 'test']:
        (out/'images'/s).mkdir(parents=True, exist_ok=True)
        (out/'labels'/s).mkdir(parents=True, exist_ok=True)
    pairs = all_pairs()
    random.shuffle(pairs)
    n = len(pairs)
    n_train, n_val = int(n*TRAIN_RATIO), int(n*VAL_RATIO)
    train, val, test = pairs[:n_train], pairs[n_train:n_train+n_val], pairs[n_train+n_val:]
    # train + aug
    for img, lbl in train:
        shutil.copy(img, out/'images/train'/Path(img).name)
        shutil.copy(lbl, out/'labels/train'/Path(lbl).name)
        image = cv2.imread(img)
        b, c = read_labels(lbl)
        for i in range(AUGS_PER_IMAGE):
            try:
                t = transform(image=image, bboxes=b, class_labels=c)
                if not t['bboxes']: continue
                aug_img = f"{Path(img).stem}_aug{i+1}.jpg"
                aug_lbl = f"{Path(lbl).stem}_aug{i+1}.txt"
                cv2.imwrite(str(out/'images/train'/aug_img), t['image'])
                save_labels(out/'labels/train'/aug_lbl, t['bboxes'], t['class_labels'])
            except Exception:
                continue
    # val
    for img, lbl in val:
        shutil.copy(img, out/'images/val'/Path(img).name)
        shutil.copy(lbl, out/'labels/val'/Path(lbl).name)
    # test
    for img, lbl in test:
        shutil.copy(img, out/'images/test'/Path(img).name)
        shutil.copy(lbl, out/'labels/test'/Path(lbl).name)
    print(f"Done! Train: {len(train)}, Val: {len(val)}, Test: {len(test)}. Aug x{AUGS_PER_IMAGE}") 