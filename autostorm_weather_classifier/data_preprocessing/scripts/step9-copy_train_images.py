import os
import json
import shutil

def copy_train_images(train_ann_file, image_dir, output_image_dir):
    os.makedirs(output_image_dir, exist_ok=True)

    with open(train_ann_file, 'r') as f:
        annotations = json.load(f)

    copied = 0
    missing = 0
    for ann in annotations:
        image_name = ann["image_name"]
        src = os.path.join(image_dir, image_name)
        dst = os.path.join(output_image_dir, image_name)

        if os.path.exists(src):
            shutil.copy(src, dst)
            copied += 1
        else:
            print(f"❗ Missing: {image_name}")
            missing += 1

    print(f"\n Copied {copied} images to {output_image_dir}")
    

# === USAGE ===
copy_train_images(
    train_ann_file="../annotations/train_annotations.json",
    image_dir="../dataset/1k/img", 
    output_image_dir="../dataset/1k/train/img"
)
