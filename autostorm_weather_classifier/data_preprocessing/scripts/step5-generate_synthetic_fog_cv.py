import os
import cv2
import json
import numpy as np
import random

def apply_realistic_fog(image, alpha_range=(0.4, 0.7), blur_ksize=(31, 31), brightness=0.75):
    h, w = image.shape[:2]

    # Step 1: Create depth-based alpha mask (stronger fog at the top of image)
    alpha_map = np.tile(np.linspace(alpha_range[0], alpha_range[1], h).reshape(h, 1), (1, w))
    fog_layer = np.full_like(image, 255)

    # Step 2: Blend original with fog layer using alpha mask
    foggy = (image * (1 - alpha_map[..., None]) + fog_layer * alpha_map[..., None]).astype(np.uint8)

    # Step 3: Apply blur
    foggy = cv2.GaussianBlur(foggy, blur_ksize, 0)

    # Step 4: Adjust brightness
    foggy = np.clip(foggy * brightness, 0, 255).astype(np.uint8)

    return foggy

def generate_synthetic_foggy_images(
    input_img_dir,
    input_ann_file,
    output_img_dir,
    output_ann_file,
    fog_count=5
):
    os.makedirs(output_img_dir, exist_ok=True)

    with open(input_ann_file, "r") as f:
        all_annotations = json.load(f)

    foggy_annotations = []
    selected = [a for a in all_annotations if a["weather"] in {"overcast", "partly cloudy"}]
    selected = random.sample(selected, min(fog_count, len(selected)))
    
    for ann in selected:
        orig_img_path = os.path.join(input_img_dir, ann["image_name"])
        if not os.path.exists(orig_img_path):
            continue

        image = cv2.imread(orig_img_path)
        if image is None:
            continue

        foggy_image = apply_realistic_fog(image)

        new_name = f"foggy_{ann['image_name']}"
        foggy_path = os.path.join(output_img_dir, new_name)
        cv2.imwrite(foggy_path, foggy_image)

        # Copy and update annotation
        foggy_ann = ann.copy()
        foggy_ann["image_name"] = new_name
        foggy_ann["weather"] = "foggy"
        foggy_annotations.append(foggy_ann)

    # Save foggy annotations
    with open(output_ann_file, 'w') as f:
        json.dump(foggy_annotations, f, indent=2)

    print(f"\ Generated {len(foggy_annotations)} synthetic foggy images.")
    print(f" Annotations saved to '{output_ann_file}'")


generate_synthetic_foggy_images(
    input_img_dir="../../bdd100k/25k/train/img",
    input_ann_file="../annotations/combined_balanced_annotations.json",
    output_img_dir="../dataset/1k/foggy",
    output_ann_file="../annotations/synthetic_foggy_annotations_cv.json",
    fog_count=100
)

def simulate_fog(image, depth_map, beta=0.08):
    """
    Simulates fog based on the exponential atmospheric scattering model.
    
    Parameters:
    - image: normalized RGB image (H, W, 3), float32 in [0, 1]
    - depth_map: depth map (H, W), float32
    - beta: atmospheric attenuation coefficient
    
    Returns:
    - foggy image (H, W, 3), float32 in [0, 1]
    """
    # Normalize depth to [0, 1]
    depth_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-6)

    # Calculate transmission map: T(x) = exp(-beta * depth)
    transmission = np.exp(-beta * depth_norm)

    # Fog color (white atmospheric light)
    A = np.ones_like(image)  # white light

    # Apply atmospheric scattering model: I = I * T + A * (1 - T)
    transmission = np.expand_dims(transmission, axis=2)  # for broadcasting
    foggy = image * transmission + A * (1 - transmission)

    return np.clip(foggy, 0, 1)
