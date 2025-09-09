import os
import random
from PIL import Image, ImageEnhance, ImageOps

# Paths
dataset_dir = "dataset/Flicker8k_Dataset"
caption_file = "dataset/Flickr8k_text/Flickr8k.token.txt"
train_images_file = "dataset/Flickr8k_text/Flickr_8k.trainImages.txt"
augmented_log_file = "dataset/augmentation_log.txt"
NUM_AUG = 2

# def augment_image(image, num_aug=NUM_AUG):
#     all_augmentations = [
#         ImageOps.mirror(image),  # Horizontal flip
#         ImageOps.autocontrast(image),  # Autocontrast
#         ImageEnhance.Brightness(image).enhance(1.2),  # Brightness increase
#         ImageEnhance.Sharpness(image).enhance(2.0),  # Sharpen
#     ]
#     return all_augmentations[:num_aug]


import imgaug.augmenters as iaa
import numpy as np
from PIL import Image

# Define advanced augmentation sequence
imgaug_seq = iaa.Sequential([
    iaa.Fliplr(0.5),  # Horizontal flip with 50% chance
    iaa.Affine(
        rotate=(-15, 15),               # Random rotation
        translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}  # Slight translation
    ),
    iaa.GaussianBlur(sigma=(0, 1.0)),  # Random blur
    iaa.LinearContrast((0.75, 1.5)),   # Contrast change
    iaa.AdditiveGaussianNoise(scale=(0, 0.03 * 255)),  # Add noise
    iaa.Multiply((0.8, 1.2)),          # Brightness scaling
    iaa.Sometimes(0.3, iaa.Grayscale(alpha=(0.0, 1.0))) # Convert to grayscale (randomly)
])

# Function to apply N augmentations
def augment_image(image_pil, num_aug=2):
    image_np = np.array(image_pil)
    augmented_images = []

    for _ in range(num_aug):
        aug_np = imgaug_seq(image=image_np)
        aug_pil = Image.fromarray(aug_np)
        augmented_images.append(aug_pil)

    return augmented_images


def main(flag = True):
    if not flag:
        return True
    
    try:
        # Load train image names
        with open(train_images_file, 'r') as f:
            train_images = set(line.strip() for line in f.readlines())
            print(f"Loaded {len(train_images)} original train images.")

        # Load captions
        captions_dict = {}
        with open(caption_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    img_id = parts[0].split('#')[0]
                    caption = parts[1]
                    if img_id in train_images:
                        captions_dict.setdefault(img_id, []).append(caption)

        # Run augmentation and save
        augmented_names = []

        with open(caption_file, 'a') as cap_file, open(train_images_file, 'a') as img_file, open(augmented_log_file, 'w') as log_file:
            for img_name in train_images:
                img_path = os.path.join(dataset_dir, img_name)
                if not os.path.exists(img_path):
                    continue

                try:
                    image = Image.open(img_path).convert("RGB")
                    aug_images = augment_image(image, NUM_AUG)

                    for idx, aug_img in enumerate(aug_images):
                        new_name = img_name.replace(".jpg", f"_aug{idx+1}.jpg")
                        new_path = os.path.join(dataset_dir, new_name)
                        aug_img.save(new_path)
                        augmented_names.append(new_name)

                        # Add same captions with new image name
                        for i, caption in enumerate(captions_dict[img_name]):
                            cap_file.write(f"{new_name}#{i}\t{caption}\n")

                        # Add to train list
                        img_file.write(new_name + "\n")

                        # Log
                        log_file.write(f"Augmented: {img_name} -> {new_name} | Captions: {len(captions_dict[img_name])}\n")

                except Exception as e:
                    log_file.write(f"Failed: {img_name} | Error: {str(e)}\n")
        return True

    except Exception as e:
        print(f"Script failed: {str(e)}")
        return False

# To run directly
if __name__ == "__main__":
    success = main()
    print("Success" if success else "Failed")
