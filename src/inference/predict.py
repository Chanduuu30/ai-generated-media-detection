from pathlib import Path
import argparse
import sys

import cv2
import joblib
import numpy as np
from PIL import Image

import torch
from torchvision import models


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# MODEL PATHS
# ============================================================

IMAGE_MODEL_PATH = (
    PROJECT_ROOT / "models/final_random_forest.pkl"
)

VIDEO_MODEL_PATH = (
    PROJECT_ROOT / "models/video/video_all_frame_svm.pkl"
)


# ============================================================
# CONSTANTS
# ============================================================

VIDEO_FRAMES = 12
VIDEO_FEATURES = 2304


# ============================================================
# IMAGE FEATURE EXTRACTION
# ============================================================

def extract_image_features(image_path):

    # Use the EXACT functions used by the
    # original feature-extraction pipeline.

    from src.features.image_features import (
        extract_features as extract_color_features
    )

    from src.features.texture_features import (
        extract_features as extract_texture_features
    )

    from src.features.frequency_features import (
        extract_features as extract_frequency_features
    )

    print(
        "\n===== EXTRACTING IMAGE FEATURES ====="
    )

    color = np.asarray(
        extract_color_features(image_path),
        dtype=np.float32
    ).flatten()

    texture = np.asarray(
        extract_texture_features(image_path),
        dtype=np.float32
    ).flatten()

    frequency = np.asarray(
        extract_frequency_features(image_path),
        dtype=np.float32
    ).flatten()

    print(
        f"Color features: {len(color)}"
    )

    print(
        f"Texture features: {len(texture)}"
    )

    print(
        f"Frequency features: {len(frequency)}"
    )

    features = np.concatenate(
        [
            color,
            texture,
            frequency
        ]
    )

    print(
        f"Combined feature count: "
        f"{len(features)}"
    )

    return features.reshape(1, -1)


# ============================================================
# VIDEO FRAME SAMPLING
# ============================================================

def sample_video_frames(
    video_path,
    num_frames=VIDEO_FRAMES
):

    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():

        raise ValueError(
            f"Unable to open video:\n"
            f"{video_path}"
        )

    total_frames = int(
        capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    if total_frames <= 0:

        capture.release()

        raise ValueError(
            "Video contains no readable frames."
        )

    indices = np.linspace(
        0,
        total_frames - 1,
        num=min(
            num_frames,
            total_frames
        ),
        dtype=int
    )

    frames = []

    for index in indices:

        capture.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(index)
        )

        success, frame = capture.read()

        if not success:
            continue

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        frames.append(frame)

    capture.release()

    if not frames:

        raise ValueError(
            "No frames could be extracted."
        )

    return frames


# ============================================================
# MOBILENET FEATURE EXTRACTOR
# ============================================================

def load_mobilenet():

    print(
        "\n===== LOADING MOBILENETV3-SMALL ====="
    )

    weights = (
        models.MobileNet_V3_Small_Weights.DEFAULT
    )

    model = models.mobilenet_v3_small(
        weights=weights
    )

    extractor = torch.nn.Sequential(
        model.features,
        model.avgpool,
        torch.nn.Flatten()
    )

    extractor.eval()

    for parameter in extractor.parameters():
        parameter.requires_grad = False

    return (
        extractor,
        weights.transforms()
    )


# ============================================================
# VIDEO FEATURE EXTRACTION
# ============================================================

def extract_video_features(video_path):

    frames = sample_video_frames(
        video_path
    )

    print(
        f"\nFrames extracted: {len(frames)}"
    )

    model, transform = load_mobilenet()

    tensors = []

    for frame in frames:

        image = Image.fromarray(
            frame
        )

        tensors.append(
            transform(image)
        )

    batch = torch.stack(
        tensors
    )

    print(
        "Extracting MobileNet features..."
    )

    with torch.no_grad():

        frame_features = model(
            batch
        )

    frame_features = (
        frame_features
        .cpu()
        .numpy()
    )

    print(
        f"Frame feature shape: "
        f"{frame_features.shape}"
    )

    # Same aggregation used when the
    # all-frame video dataset was created.

    mean_features = frame_features.mean(
        axis=0
    )

    std_features = frame_features.std(
        axis=0
    )

    max_features = frame_features.max(
        axis=0
    )

    min_features = frame_features.min(
        axis=0
    )

    combined = np.concatenate(
        [
            mean_features,
            std_features,
            max_features,
            min_features
        ]
    )

    print(
        f"Video feature count: "
        f"{len(combined)}"
    )

    if len(combined) != VIDEO_FEATURES:

        raise ValueError(
            f"Expected {VIDEO_FEATURES} "
            f"video features, got "
            f"{len(combined)}"
        )

    return combined.reshape(1, -1)


# ============================================================
# IMAGE PREDICTION
# ============================================================

def predict_image(image_path):

    print(
        "\n========================================"
    )

    print(
        "        IMAGE AI-DETECTION"
    )

    print(
        "========================================"
    )

    print(
        f"Input: {image_path}"
    )

    model = joblib.load(
        IMAGE_MODEL_PATH
    )

    features = extract_image_features(
        image_path
    )

    print(
        f"\nModel expects: "
        f"{model.n_features_in_} features"
    )

    print(
        f"Features supplied: "
        f"{features.shape[1]}"
    )

    if features.shape[1] != model.n_features_in_:

        raise ValueError(
            "\nIMAGE MODEL FEATURE MISMATCH\n"
            f"Model expects "
            f"{model.n_features_in_}, "
            f"but extraction produced "
            f"{features.shape[1]}.\n\n"
            "We will need to reproduce the exact "
            "combine_all_features.py feature "
            "selection before inference."
        )

    prediction = model.predict(
        features
    )[0]

    probabilities = model.predict_proba(
        features
    )[0]

    real_probability = probabilities[0]

    synthetic_probability = probabilities[1]

    if prediction == 1:

        label = "SYNTHETIC"
        confidence = synthetic_probability

    else:

        label = "REAL"
        confidence = real_probability

    print(
        "\n===== RESULT ====="
    )

    print(
        f"Prediction: {label}"
    )

    print(
        f"Confidence: "
        f"{confidence * 100:.2f}%"
    )

    print(
        f"Real probability: "
        f"{real_probability * 100:.2f}%"
    )

    print(
        f"Synthetic probability: "
        f"{synthetic_probability * 100:.2f}%"
    )

    print(
        "Model: Random Forest"
    )


# ============================================================
# VIDEO PREDICTION
# ============================================================

def predict_video(video_path):

    print(
        "\n========================================"
    )

    print(
        "        VIDEO AI-DETECTION"
    )

    print(
        "========================================"
    )

    print(
        f"Input: {video_path}"
    )

    model = joblib.load(
        VIDEO_MODEL_PATH
    )

    features = extract_video_features(
        video_path
    )

    print(
        f"\nModel expects: "
        f"{model.n_features_in_} features"
    )

    prediction = model.predict(
        features
    )[0]

    probabilities = model.predict_proba(
        features
    )[0]

    real_probability = probabilities[0]

    synthetic_probability = probabilities[1]

    if prediction == 1:

        label = "SYNTHETIC"
        confidence = synthetic_probability

    else:

        label = "REAL"
        confidence = real_probability

    print(
        "\n===== RESULT ====="
    )

    print(
        f"Prediction: {label}"
    )

    print(
        f"Confidence: "
        f"{confidence * 100:.2f}%"
    )

    print(
        f"Real probability: "
        f"{real_probability * 100:.2f}%"
    )

    print(
        f"Synthetic probability: "
        f"{synthetic_probability * 100:.2f}%"
    )

    print(
        "Model: MobileNetV3-Small + Linear SVM"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "AI-generated media detection"
        )
    )

    group = parser.add_mutually_exclusive_group(
        required=True
    )

    group.add_argument(
        "--image",
        type=str,
        help="Path to image"
    )

    group.add_argument(
        "--video",
        type=str,
        help="Path to video"
    )

    args = parser.parse_args()

    if args.image:

        path = (
            Path(args.image)
            .expanduser()
            .resolve()
        )

        if not path.exists():

            print(
                f"ERROR: Image not found:\n{path}"
            )

            sys.exit(1)

        try:

            predict_image(path)

        except Exception as error:

            print(
                "\nERROR during image analysis:"
            )

            print(error)

            sys.exit(1)

    elif args.video:

        path = (
            Path(args.video)
            .expanduser()
            .resolve()
        )

        if not path.exists():

            print(
                f"ERROR: Video not found:\n{path}"
            )

            sys.exit(1)

        try:

            predict_video(path)

        except Exception as error:

            print(
                "\nERROR during video analysis:"
            )

            print(error)

            sys.exit(1)


if __name__ == "__main__":
    main()