"""
Sample Dataset Generator for Leaf Disease Detection
Creates a populated dataset directory structure with realistic synthetic plant leaf images
so the model training and prediction app can be executed and tested immediately out-of-the-box.
"""

import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")
SPLITS = ["train", "validation", "test"]

CATEGORIES = [
    "Tomato_Early_Blight",
    "Tomato_Late_Blight",
    "Tomato_Leaf_Spot",
    "Tomato_Powdery_Mildew",
    "Tomato_Healthy",
    "Potato_Early_Blight",
    "Potato_Late_Blight",
    "Potato_Healthy",
    "Apple_Scab",
    "Apple_Black_Rot",
    "Apple_Healthy",
    "Corn_Common_Rust",
    "Corn_Healthy"
]

# Samples count per split category
SAMPLES_PER_SPLIT = {
    "train": 40,
    "validation": 10,
    "test": 10
}

def create_synthetic_leaf(category, width=224, height=224):
    """
    Synthesizes a realistic plant leaf image with category-specific health patterns,
    spots, mold textures, or clean leaf foliage.
    """
    # Base leaf background color (varied shades of green)
    if "Healthy" in category:
        bg_color = (random.randint(35, 60), random.randint(140, 190), random.randint(40, 75))
    elif "Powdery" in category:
        bg_color = (random.randint(50, 80), random.randint(110, 150), random.randint(50, 80))
    else:
        bg_color = (random.randint(45, 75), random.randint(100, 145), random.randint(35, 60))

    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    # Draw leaf veins
    vein_color = (bg_color[0] + 25, bg_color[1] + 30, bg_color[2] + 20)
    # Main central vein
    draw.line([(width // 2, 10), (width // 2, height - 10)], fill=vein_color, width=4)
    # Lateral veins
    for y in range(30, height - 30, 25):
        draw.line([(width // 2, y), (20, y - 15)], fill=vein_color, width=2)
        draw.line([(width // 2, y), (width - 20, y - 15)], fill=vein_color, width=2)

    # Add disease specific features
    if "Early_Blight" in category:
        # Concentric dark ring spots
        for _ in range(random.randint(4, 8)):
            cx, cy = random.randint(40, width - 40), random.randint(40, height - 40)
            r = random.randint(12, 22)
            # Yellow halo
            draw.ellipse([cx - r - 4, cy - r - 4, cx + r + 4, cy + r + 4], fill=(180, 170, 30))
            # Dark brown lesion center
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(60, 35, 15))
            # Inner concentric ring
            draw.ellipse([cx - r//2, cy - r//2, cx + r//2, cy + r//2], fill=(35, 20, 10))

    elif "Late_Blight" in category:
        # Large dark water-soaked blotches
        for _ in range(random.randint(2, 5)):
            cx, cy = random.randint(30, width - 30), random.randint(30, height - 30)
            rx, ry = random.randint(20, 45), random.randint(15, 35)
            draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=(40, 45, 30))
            # Mold fuzz edge
            draw.ellipse([cx - rx + 5, cy - ry + 5, cx + rx - 5, cy + ry - 5], fill=(70, 50, 35))

    elif "Leaf_Spot" in category or "Scab" in category:
        # Small circular dark spots
        for _ in range(random.randint(12, 25)):
            cx, cy = random.randint(20, width - 20), random.randint(20, height - 20)
            r = random.randint(3, 8)
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(45, 25, 15))

    elif "Powdery_Mildew" in category:
        # White talcum powder-like patches
        for _ in range(random.randint(5, 10)):
            cx, cy = random.randint(25, width - 25), random.randint(25, height - 25)
            rx, ry = random.randint(15, 30), random.randint(10, 25)
            draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=(220, 225, 220))

    elif "Rust" in category or "Black_Rot" in category:
        # Cinnamon reddish pustules or dark rot spots
        color = (160, 60, 20) if "Rust" in category else (25, 20, 20)
        for _ in range(random.randint(8, 16)):
            cx, cy = random.randint(30, width - 30), random.randint(30, height - 30)
            r = random.randint(4, 10)
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)

    elif "Bacterial_Spot" in category:
        # Small angular water-soaked dark spots
        for _ in range(random.randint(10, 20)):
            cx, cy = random.randint(25, width - 25), random.randint(25, height - 25)
            r = random.randint(4, 7)
            draw.rectangle([cx - r, cy - r, cx + r, cy + r], fill=(30, 35, 25))

    # Apply subtle smooth blur to simulate camera focus
    img_blurred = img.filter(ImageFilter.GaussianBlur(radius=0.5))
    return img_blurred

def generate_dataset():
    """Generates synthetic dataset structure across splits and categories."""
    print("🌿 Starting sample plant leaf dataset generation...")
    total_generated = 0

    for split in SPLITS:
        num_samples = SAMPLES_PER_SPLIT[split]
        for category in CATEGORIES:
            cat_dir = os.path.join(DATASET_DIR, split, category)
            os.makedirs(cat_dir, exist_ok=True)

            for i in range(num_samples):
                img = create_synthetic_leaf(category)
                file_name = f"{category.lower()}_{split}_{i+1:03d}.jpg"
                file_path = os.path.join(cat_dir, file_name)
                img.save(file_path, quality=92)
                total_generated += 1

    print(f"✅ Dataset generation complete! Created {total_generated} images across {len(CATEGORIES)} categories in '{DATASET_DIR}'.")

if __name__ == "__main__":
    generate_dataset()
