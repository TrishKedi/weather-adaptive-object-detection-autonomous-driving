import json
import os

def convert_to_coco(input_json, output_json):
    with open(input_json, 'r') as f:
        data = json.load(f)

    images = []
    annotations = []
    categories = {}
    ann_id = 1
    img_id = 1

    for item in data:
        image_name = item['image_name']
        height = item['size']['height']
        width = item['size']['width']

        images.append({
            "id": img_id,
            "file_name": image_name,
            "height": height,
            "width": width
        })

        for obj in item['objects']:
            if obj['geometryType'] != 'rectangle':
                continue  # Skip non-bounding-box shapes

            class_name = obj['classTitle']
            if class_name not in categories:
                categories[class_name] = len(categories) + 1

            category_id = categories[class_name]

            x1, y1 = obj['points']['exterior'][0]
            x2, y2 = obj['points']['exterior'][1]

            x = float(min(x1, x2))
            y = float(min(y1, y2))
            w = float(abs(x2 - x1))
            h = float(abs(y2 - y1))

            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": category_id,
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0
            })
            ann_id += 1

        img_id += 1

    coco_format = {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": cid, "name": cname}
            for cname, cid in categories.items()
        ]
    }

    with open(output_json, 'w') as f:
        json.dump(coco_format, f, indent=2)

    print(f" Converted to COCO format → '{output_json}'")
    print(f" Images: {len(images)} | Annotations: {len(annotations)} | Categories: {len(categories)}")

input_json = "../annotations/train_annotations.json"
output_json = "../dataset/1k/train/coco_annotations.json"
convert_to_coco(input_json, output_json)

# categories = ['clear', 'foggy', 'overcast', 'partly cloudy', 'rainy', 'snowy']

# for cat in categories:
#     input_json = f"../dataset/1k/val/{cat}/annotations.json"
#     output_json = f"../dataset/1k/val/{cat}/coco_annotations.json"
#     convert_to_coco(input_json, output_json)

