import json
import os
import shutil
from collections import defaultdict

def split_by_weather(input_ann_file, image_dir, output_root):
    with open(input_ann_file, "r") as f:
        anns = json.load(f)

    grouped = defaultdict(list)
    for entry in anns:
        weather = entry.get("weather", "undefined").lower()
        if weather == "undefined":
            continue
        grouped[weather].append(entry)

    for weather, entries in grouped.items():
        weather_dir = os.path.join(output_root, weather)
        os.makedirs(os.path.join(weather_dir, "images"), exist_ok=True)

        # Copy images and save annotations
        for entry in entries:
            src = os.path.join(image_dir, entry["image_name"])
            dst = os.path.join(weather_dir, "images", entry["image_name"])
            if os.path.exists(src):
                shutil.copy(src, dst)

        ann_out = os.path.join(weather_dir, "annotations.json")
        with open(ann_out, "w") as f:
            json.dump(entries, f, indent=2)

        print(f"✅ {weather.upper()}: {len(entries)} images saved to {weather_dir}")

# === USAGE ===
split_by_weather(
    input_ann_file="../annotations/test_annotations.json",
    image_dir="../dataset/1k/img",
    output_root="../dataset/1k/test/"
)


