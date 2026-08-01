import json
import random

def split_annotations(input_file, train_file, val_file, test_file=None, val_ratio=0.1, test_ratio=0.1, seed=42):
    with open(input_file, "r") as f:
        data = json.load(f)

    random.seed(seed)
    random.shuffle(data)

    total = len(data)
    val_count = int(total * val_ratio)
    test_count = int(total * test_ratio) if test_file else 0
    train_count = total - val_count - test_count

    train_data = data[:train_count]
    val_data = data[train_count:train_count + val_count]
    test_data = data[train_count + val_count:] if test_file else []

    with open(train_file, 'w') as f:
        json.dump(train_data, f, indent=2)
    with open(val_file, 'w') as f:
        json.dump(val_data, f, indent=2)
    if test_file:
        with open(test_file, 'w') as f:
            json.dump(test_data, f, indent=2)

    print(f" Split complete:")
    print(f"   Train: {len(train_data)}")
    print(f"   Val:   {len(val_data)}")
    print(f"   Test:  {len(test_data)}")

# === USAGE ===
split_annotations(
    input_file="../annotations/final_annotations.json",
    train_file="../annotations/train_annotations.json",
    val_file="../annotations/val_annotations.json",
    test_file="../annotations/test_annotations.json", 
    val_ratio=0.1,
    test_ratio=0.1
)
