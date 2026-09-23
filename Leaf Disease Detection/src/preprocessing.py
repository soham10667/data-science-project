import os
import io
import numpy as np
from PIL import Image
import cv2

TARGET_SIZE = (224, 224)
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png'}

# PyTorch ImageNet Normalization Constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def validate_image(image_input):
    """
    Validates an uploaded image or file path.
    Checks file existence/readability, format, and corruption.
    
    Args:
        image_input: File path (str), BytesIO, or UploadedFile object.
        
    Returns:
        tuple: (is_valid (bool), message_or_img (PIL.Image or str))
    """
    try:
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                return False, f"File path does not exist: {image_input}"
            ext = os.path.splitext(image_input)[1].lower()
            if ext not in ALLOWED_EXTENSIONS:
                return False, f"Unsupported file extension '{ext}'. Allowed: JPG, JPEG, PNG."
            img = Image.open(image_input)
        elif hasattr(image_input, 'read'):
            image_bytes = image_input.read()
            if hasattr(image_input, 'seek'):
                image_input.seek(0)
            img = Image.open(io.BytesIO(image_bytes))
        else:
            return False, "Invalid image input format."

        # Verify image format & mode
        img.verify()
        
        # Re-open after verify()
        if isinstance(image_input, str):
            img = Image.open(image_input)
        else:
            if hasattr(image_input, 'seek'):
                image_input.seek(0)
            img = Image.open(io.BytesIO(image_bytes))

        # Convert to RGB mode if RGBA or Grayscale
        if img.mode != "RGB":
            img = img.convert("RGB")
            
        return True, img

    except Exception as e:
        return False, f"Corrupted or invalid image file: {str(e)}"

def load_and_preprocess_image(image_input, target_size=TARGET_SIZE):
    """
    Loads, resizes, and normalizes an image into a PyTorch Tensor batch format (1, 3, 224, 224).
    """
    if isinstance(image_input, str) or hasattr(image_input, 'read'):
        is_valid, result = validate_image(image_input)
        if not is_valid:
            raise ValueError(result)
        img = result
    elif isinstance(image_input, Image.Image):
        img = image_input.convert("RGB") if image_input.mode != "RGB" else image_input
    else:
        raise ValueError("Unsupported input type for preprocessing.")

    try:
        import torch
        from torchvision import transforms
        
        transform_pipeline = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        
        img_tensor = transform_pipeline(img)
        # Add batch dimension: (1, 3, 224, 224)
        img_batch = img_tensor.unsqueeze(0)
        return img_batch
    except ImportError:
        # Fallback NumPy array format
        img_resized = img.resize(target_size, Image.Resampling.BILINEAR)
        img_array = np.array(img_resized, dtype=np.float32) / 255.0
        return np.expand_dims(img_array, axis=0)

def get_train_transforms(target_size=TARGET_SIZE):
    """Returns PyTorch data augmentation transform pipeline for training dataset."""
    try:
        from torchvision import transforms
        return transforms.Compose([
            transforms.Resize(target_size),
            transforms.RandomRotation(25),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(target_size[0], scale=(0.8, 1.0)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    except ImportError:
        return None

def get_eval_transforms(target_size=TARGET_SIZE):
    """Returns PyTorch transform pipeline for validation/testing datasets."""
    try:
        from torchvision import transforms
        return transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    except ImportError:
        return None
