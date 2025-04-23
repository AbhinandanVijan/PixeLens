import os
import random
from PIL import Image, ImageEnhance, ImageOps

# Paths
dataset_dir = "dataset/Flicker8k_Dataset"
caption_file = "dataset/Flickr8k_text/Flickr8k.lemma.token.txt"
train_images_file = "dataset/Flickr8k_text/Flickr_8k.trainImages.txt"
augmented_log_file = "dataset/augmentation_log.txt"
NUM_AUG = 2

# Load train image names
with open(train_images_file, 'r') as f:
    train_images = set(line.strip() for line in f.readlines())
    print(f"Loaded {len(train_images)} train images.")

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

def augment_image(image, num_aug=NUM_AUG):
    all_augmentations = [
        ImageOps.mirror(image),  # Horizontal flip
        ImageOps.autocontrast(image),  # Autocontrast
        ImageEnhance.Brightness(image).enhance(1.2),  # Brightness increase
        ImageEnhance.Sharpness(image).enhance(2.0),  # Sharpen
    ]
    return all_augmentations[:num_aug]


# Run augmentation and save
augmented_names = []

with open(caption_file, 'a') as cap_file, open(train_images_file, 'a') as img_file, open(augmented_log_file, 'w') as log_file:
    for img_name in train_images:
        img_path = os.path.join(dataset_dir, img_name)
        if not os.path.exists(img_path):
            continue

        try:
            image = Image.open(img_path).convert("RGB")
            aug_images = augment_image(image,NUM_AUG)

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







# import os
# from PIL import Image
# from torchvision import transforms
# from datetime import datetime

# # Configuration
# image_dir = "dataset/Flicker8k_Dataset"
# caption_file = "dataset/Flickr8k_text/Flickr8k.lemma.token.txt"
# log_file = "augmentation_log.txt"
# NUM_AUG = 2

# # Augmentation Pipeline
# augment = transforms.Compose([
#     transforms.RandomHorizontalFlip(p=1.0),
#     transforms.RandomRotation(15),
#     transforms.ColorJitter(brightness=0.2, contrast=0.2),
#     transforms.Resize((224, 224))
# ])

# # Load original captions
# caption_dict = {}
# with open(caption_file, 'r') as f:
#     for line in f:
#         image_id_full, caption = line.strip().split("\t")
#         image_id = image_id_full.split("#")[0]
#         caption_dict.setdefault(image_id, []).append(caption)

# # Prepare new caption lines and log lines
# new_caption_lines = []
# log_lines = []
# log_lines.append(f"\n# Log started at {datetime.now()}\n")

# # Begin augmentation
# for image_name, captions in caption_dict.items():
#     image_path = os.path.join(image_dir, image_name)
#     if not os.path.exists(image_path):
#         log_lines.append(f"{image_name} - ❌ NOT FOUND")
#         continue

#     try:
#         image = Image.open(image_path).convert("RGB")
#     except:
#         log_lines.append(f"{image_name} - ⚠️ FAILED TO LOAD")
#         continue

#     for i in range(1, NUM_AUG + 1):
#         aug_img = augment(image)
#         new_name = image_name.replace(".jpg", f"_aug{i}.jpg")
#         new_path = os.path.join(image_dir, new_name)
#         try:
#             aug_img.save(new_path)
#             for idx, caption in enumerate(captions):
#                 new_caption_lines.append(f"{new_name}#{idx}\t{caption}")
#             log_lines.append(f"{image_name} → {new_name} - ✅ Image and {len(captions)} captions saved")
#         except Exception as e:
#             log_lines.append(f"{image_name} → {new_name} - ❌ Save Failed: {str(e)}")

# # Append new captions to token file
# with open(caption_file, "a") as f:
#     for line in new_caption_lines:
#         f.write(line + "\n")

# # Write log file
# with open(log_file, "a") as log:
#     for line in log_lines:
#         log.write(line + "\n")

# print(f"✅ Done. Augmented {len(new_caption_lines)} captions across {len(log_lines)-1} images.")
# print(f"📄 Log saved to {log_file}")
