import os
import json
from collections import Counter

"""
Extracts weather categories/classes in bdd100k dataset
"""
def extract_weather_tags(json_dir):
    weather_values = []

    for filename in os.listdir(json_dir):
        if filename.endswith('.json'):
            with open(os.path.join(json_dir, filename), 'r') as f:
                data = json.load(f)
                tags = data.get("tags", [])
                for tag in tags:
                    if tag.get("name") == "weather":
                        weather_values.append(tag.get("value"))

    return Counter(weather_values)


json_folder1 = "../../bdd100k/train/ann"
weather_counter1 = extract_weather_tags(json_folder1)
json_folder2 = "../../bdd100k/test/ann"
weather_counter2 = extract_weather_tags(json_folder2)
json_folder3= "../../bdd100k/val/ann"
weather_counter3 = extract_weather_tags(json_folder3)


print("train: ")
print(weather_counter1)

print("test: ")
print(weather_counter2)

print("val: ")
print(weather_counter3)



    
