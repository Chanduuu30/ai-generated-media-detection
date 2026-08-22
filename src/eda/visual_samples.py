from pathlib import Path
import random

import matplotlib.pyplot as plt
from PIL import Image

DATASET_DIR = Path("data/raw/mediaeval/ITW-SM")

REAL_DIR = DATASET_DIR / "0_real"
SYNTHETIC_DIR = DATASET_DIR / "1_fake"

NUM_SAMPLES = 6
RANDOM_SEED = 42

random.seed(RANDOM_SEED)


def get_valid_images(folder):
    """Return image files that can be opened successfully."""
    valid_images = []

    for image_path in folder.iterdir():
        if not image_path.is_file():
            continue

        try:
            with Image.open(image_path) as image:
                image.verify()

            valid_images.append(image_path)

        except Exception:
            continue

    return valid_images


real_images = get_valid_images(REAL_DIR)
synthetic_images = get_valid_images(SYNTHETIC_DIR)

real_samples = random.sample(real_images, NUM_SAMPLES)
synthetic_samples = random.sample(synthetic_images, NUM_SAMPLES)

fig, axes = plt.subplots(2, NUM_SAMPLES, figsize=(18, 7))

for index, image_path in enumerate(real_samples):
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        axes[0, index].imshow(image)

    axes[0, index].set_title(f"Real\n{image_path.name}")
    axes[0, index].axis("off")


for index, image_path in enumerate(synthetic_samples):
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        axes[1, index].imshow(image)

    axes[1, index].set_title(f"Synthetic\n{image_path.name}")
    axes[1, index].axis("off")


fig.suptitle(
    "MediaEval Dataset - Visual Sample Inspection",
    fontsize=16
)

plt.tight_layout()

output_path = Path("data/processed/visual_samples.png")
output_path.parent.mkdir(parents=True, exist_ok=True)

plt.savefig(output_path, dpi=150, bbox_inches="tight")
plt.show()

print(f"Real images available: {len(real_images)}")
print(f"Synthetic images available: {len(synthetic_images)}")
print(f"Visualization saved to: {output_path}")