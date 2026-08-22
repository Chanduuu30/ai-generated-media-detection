from pathlib import Path

import cv2
import numpy as np
from skimage.feature import local_binary_pattern


IMAGE_SIZE = (256, 256)

LBP_POINTS = 24
LBP_RADIUS = 3


def load_and_prepare_image(image_path):
    """
    Load an image, convert it to grayscale,
    and resize while preserving aspect ratio.
    """

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None:
        raise ValueError(
            f"Unable to read image: {image_path}"
        )

    target_width, target_height = IMAGE_SIZE

    original_height, original_width = image.shape

    scale = min(
        target_width / original_width,
        target_height / original_height,
    )

    new_width = max(
        1,
        int(original_width * scale),
    )

    new_height = max(
        1,
        int(original_height * scale),
    )

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )

    canvas = np.zeros(
        (target_height, target_width),
        dtype=np.uint8,
    )

    x_offset = (
        target_width - new_width
    ) // 2

    y_offset = (
        target_height - new_height
    ) // 2

    canvas[
        y_offset:y_offset + new_height,
        x_offset:x_offset + new_width
    ] = resized

    return canvas


def extract_texture_features(image):
    """
    Extract LBP histogram features.
    """

    lbp = local_binary_pattern(
        image,
        P=LBP_POINTS,
        R=LBP_RADIUS,
        method="uniform",
    )

    number_of_bins = LBP_POINTS + 2

    histogram, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(
            0,
            number_of_bins + 1,
        ),
        range=(
            0,
            number_of_bins,
        ),
    )

    histogram = histogram.astype(
        np.float32
    )

    histogram /= histogram.sum()

    return histogram


def extract_features(image_path):
    """
    Complete texture feature extraction pipeline.
    """

    image = load_and_prepare_image(
        image_path
    )

    features = extract_texture_features(
        image
    )

    return features


if __name__ == "__main__":

    test_image = next(
        Path(
            "data/raw/mediaeval/ITW-SM/0_real"
        ).glob("*")
    )

    features = extract_features(
        test_image
    )

    print(
        "===== TEXTURE FEATURE TEST ====="
    )

    print(
        f"Image: {test_image}"
    )

    print(
        f"Feature vector shape: "
        f"{features.shape}"
    )

    print(
        f"Number of features: "
        f"{len(features)}"
    )

    print(
        f"Features: {features}"
    )