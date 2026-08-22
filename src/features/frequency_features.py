from pathlib import Path

import cv2
import numpy as np


IMAGE_SIZE = (256, 256)


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


def extract_frequency_features(image):
    """
    Extract statistical features from the
    2D Fourier magnitude spectrum.
    """

    image_float = image.astype(
        np.float32
    ) / 255.0

    # 2D Fourier transform
    fft = np.fft.fft2(image_float)

    # Shift zero frequency to the center
    fft_shifted = np.fft.fftshift(fft)

    # Magnitude spectrum
    magnitude = np.abs(fft_shifted)

    # Log transform reduces the effect of
    # extremely large frequency values.
    magnitude = np.log1p(magnitude)

    height, width = magnitude.shape

    center_y = height // 2
    center_x = width // 2

    # Create coordinate grid
    y, x = np.ogrid[:height, :width]

    distance = np.sqrt(
        (x - center_x) ** 2
        + (y - center_y) ** 2
    )

    # Define low-frequency region
    low_frequency_mask = (
        distance <= 20
    )

    # Define high-frequency region
    high_frequency_mask = (
        distance >= 80
    )

    low_energy = np.mean(
        magnitude[low_frequency_mask]
    )

    high_energy = np.mean(
        magnitude[high_frequency_mask]
    )

    high_low_ratio = (
        high_energy
        / (low_energy + 1e-8)
    )

    features = [
        np.mean(magnitude),
        np.std(magnitude),
        np.median(magnitude),
        np.max(magnitude),
        np.percentile(magnitude, 25),
        np.percentile(magnitude, 75),
        np.percentile(magnitude, 90),
        np.percentile(magnitude, 95),
        np.percentile(magnitude, 99),
        low_energy,
        high_energy,
        high_low_ratio,
    ]

    return np.array(
        features,
        dtype=np.float32,
    )


def extract_features(image_path):
    """
    Complete frequency feature extraction pipeline.
    """

    image = load_and_prepare_image(
        image_path
    )

    features = extract_frequency_features(
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
        "===== FREQUENCY FEATURE TEST ====="
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