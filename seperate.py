from pathlib import Path
import random
import shutil


DATASET_DIR = Path(r"C:\Users\Big Blue Logistics\Downloads\New folder\project-6-at-2026-08-10-06-45-fbbd56a0")

IMAGES_DIR = DATASET_DIR / "images"
LABELS_DIR = DATASET_DIR / "labels"

TRAIN_RATIO = 0.8
RANDOM_SEED = 42


def main():
    random.seed(RANDOM_SEED)

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    images = [
        image
        for image in IMAGES_DIR.iterdir()
        if image.suffix.lower() in image_extensions
    ]

    # Only keep images that have a matching label
    paired_images = []

    for image in images:
        label = LABELS_DIR / f"{image.stem}.txt"

        if label.exists():
            paired_images.append(image)
        else:
            print(f"Warning: No label found for {image.name}")

    random.shuffle(paired_images)

    split_index = int(len(paired_images) * TRAIN_RATIO)

    train_images = paired_images[:split_index]
    val_images = paired_images[split_index:]

    directories = [
        IMAGES_DIR / "train",
        IMAGES_DIR / "val",
        LABELS_DIR / "train",
        LABELS_DIR / "val",
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    for image in train_images:
        label = LABELS_DIR / f"{image.stem}.txt"

        shutil.move(image, IMAGES_DIR / "train" / image.name)
        shutil.move(label, LABELS_DIR / "train" / label.name)

    for image in val_images:
        label = LABELS_DIR / f"{image.stem}.txt"

        shutil.move(image, IMAGES_DIR / "val" / image.name)
        shutil.move(label, LABELS_DIR / "val" / label.name)

    print()
    print("Dataset split complete!")
    print(f"Total: {len(paired_images)}")
    print(f"Train: {len(train_images)}")
    print(f"Validation: {len(val_images)}")


if __name__ == "__main__":
    main()