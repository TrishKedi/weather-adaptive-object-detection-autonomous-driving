import os
import json
import random
from collections import defaultdict

def build_balanced_subset(json_dir, output_file, limits):
    grouped = defaultdict(list)

    for filename in os.listdir(json_dir):
        if not filename.endswith(".json"):
            continue
        with open(os.path.join(json_dir, filename), "r") as f:
            data = json.load(f)

        # Extract weather
        weather = "undefined"
        for tag in data.get("tags", []):
            if tag.get("name") == "weather":
                weather = tag.get("value").lower()
                break

        if weather == "undefined":
            continue

        data["weather"] = weather
        data["image_name"] = filename.replace(".json", "")
        grouped[weather].append(data)

    selected = []
    for weather, files in grouped.items():
        limit = limits.get(weather, len(files))
        sampled = random.sample(files, min(200, len(files)))
        selected.extend(sampled)
        print(f"{weather}: selected {len(sampled)} / {len(files)}")

    # Save output
    with open(output_file, "w") as f:
        json.dump(selected, f, indent=2)

    print(f"\n Saved balanced subset to '{output_file}' with {len(selected)} annotations.")


json_folder = "../../bdd100k/train/ann"
output_combined_json = "../annotations/combined_balanced_annotations.json"

# Weather-specific limits
weather_limits = {
    "clear": 5000,
    "overcast": 5000,
    "partly cloudy": 5000,

}

build_balanced_subset(json_folder, output_combined_json, weather_limits)
