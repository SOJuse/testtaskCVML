import os
from pathlib import Path

# Пути из data.yaml
TRAIN_DIR = Path('dataset_balance/images/train')
VAL_DIR = Path('dataset/images/val')
TEST_DIR = Path('dataset/images/test')

def get_basenames(img_dir):
    return set(f.stem for f in img_dir.glob('*.jpg'))

train_set = get_basenames(TRAIN_DIR)
val_set = get_basenames(VAL_DIR)
test_set = get_basenames(TEST_DIR)

val_overlap = train_set & val_set
test_overlap = train_set & test_set

print(f"Всего в train: {len(train_set)}")
print(f"Всего в val:   {len(val_set)}")
print(f"Всего в test:  {len(test_set)}")
print()
print(f"Пересечений train/val:  {len(val_overlap)}")
print(f"Пересечений train/test: {len(test_overlap)}")

if val_overlap:
    print("\nФайлы, попавшие и в train, и в val:")
    for name in sorted(val_overlap):
        print(name)
if test_overlap:
    print("\nФайлы, попавшие и в train, и в test:")
    for name in sorted(test_overlap):
        print(name)
