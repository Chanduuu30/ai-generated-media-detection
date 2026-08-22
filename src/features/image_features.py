from pathlib import Path

import cv2
import numpy as np


IMAGE_SIZE = (256, 256)


def load_image(image_path):
    """
    Load an image and convert it to RGB.
    """

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def resize_image(image):
    """
    Resize an image while preserving its aspect ratio,
    then pad it to a fixed 256x256 size.
    """

    target_width, target_height = IMAGE_SIZE

    original_height, original_width = image.shape[:2]

    scale = min(
        target_width / original_width,
        target_height / original_height,
    )

    new_width = max(1, int(original_width * scale))
    new_height = max(1, int(original_height * scale))

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )

    canvas = np.zeros(
        (target_height, target_width, 3),
        dtype=np.uint8,
    )

    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2

    canvas[
        y_offset:y_offset + new_height,
        x_offset:x_offset + new_width
    ] = resized

    return canvas


def extract_color_features(image):
    """
    Extract basic RGB and HSV color statistics.
    """

    image_float = image.astype(np.float32) / 255.0

    features = []

    # RGB mean and standard deviation
    for channel in range(3):
        channel_data = image_float[:, :, channel]

        features.append(np.mean(channel_data))
        features.append(np.std(channel_data))

    # Convert RGB to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    hsv_float = hsv.astype(np.float32)

    # Normalize HSV channels
    hsv_float[:, :, 0] /= 179.0
    hsv_float[:, :, 1] /= 255.0
    hsv_float[:, :, 2] /= 255.0

    # HSV mean and standard deviation
    for channel in range(3):
        channel_data = hsv_float[:, :, channel]

        features.append(np.mean(channel_data))
        features.append(np.std(channel_data))

    # RGB histograms
    for channel in range(3):

        histogram, _ = np.histogram(
            image_float[:, :, channel],
            bins=16,
            range=(0.0, 1.0),
        )

        histogram = histogram.astype(np.float32)

        histogram /= histogram.sum()

        features.extend(histogram)

    return np.array(features, dtype=np.float32)


def extract_features(image_path):
    """
    Complete feature extraction pipeline.
    """

    image = load_image(image_path)

    image = resize_image(image)

    features = extract_color_features(image)

    return features


if __name__ == "__main__":

    test_image = next(
        Path("data/raw/mediaeval/ITW-SM/0_real").glob("*")
    )

    features = extract_features(test_image)

    print("===== FEATURE EXTRACTION TEST =====")
    print(f"Image: {test_image}")
    print(f"Feature vector shape: {features.shape}")
    print(f"Number of features: {len(features)}")
    print(f"First 10 features: {features[:10]}")