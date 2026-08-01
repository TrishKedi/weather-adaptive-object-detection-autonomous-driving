import os
import json
import shutil

def copy_coco_images(coco_json, source_img_dir, dest_img_dir):
    with open(coco_json, 'r') as f:
        coco = json.load(f)

    os.makedirs(dest_img_dir, exist_ok=True)
    copied = 0

    for img in coco['images']:
        file_name = img['file_name']
        src = os.path.join(source_img_dir, file_name)
        dst = os.path.join(dest_img_dir, file_name)

        if os.path.exists(src):
            shutil.copy(src, dst)
            copied += 1
        else:
            print(f"⚠️ File not found: {file_name}")

    print(f"\n✅ Copied {copied} images to '{dest_img_dir}'")


coco_json_path = "../annotations/combined_coco_annotations.json"
source_images_dir = "../../bdd100k/train/img"
target_training_dir = "../dataset/1k/img"

copy_coco_images(coco_json_path, source_images_dir, target_training_dir)
