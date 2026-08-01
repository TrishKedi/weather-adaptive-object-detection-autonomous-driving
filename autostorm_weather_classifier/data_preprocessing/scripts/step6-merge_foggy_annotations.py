import json
import os

def merge_annotation_files(original_file, foggy_file, output_file):
    with open(original_file, 'r') as f:
        original = json.load(f)

    with open(foggy_file, 'r') as f:
        foggy = json.load(f)

    merged = original + foggy

    with open(output_file, 'w') as f:
        json.dump(merged, f, indent=2)

    print(f"✅ Merged {len(original)} original + {len(foggy)} foggy annotations")
    print(f"📝 Saved to: {output_file}")


merge_annotation_files(
    original_file="../annotations/combined_balanced_annotations.json",
    foggy_file="../annotations/synthetic_foggy_annotations_cv.json",
    output_file="../annotations/final_annotations.json"
)


