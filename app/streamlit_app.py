import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch

from PIL import Image
from torchvision import models


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


IMAGE_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "final_random_forest.pkl"
)

VIDEO_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "video"
    / "video_all_frame_svm.pkl"
)


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="AI-Generated Media Detector",
    page_icon="🔍",
    layout="centered",
)


# ============================================================
# LOAD IMAGE MODEL
# ============================================================

@st.cache_resource
def load_image_model():

    if not IMAGE_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Image model not found:\n{IMAGE_MODEL_PATH}"
        )

    return joblib.load(
        IMAGE_MODEL_PATH
    )


# ============================================================
# LOAD VIDEO MODEL
# ============================================================

@st.cache_resource
def load_video_model():

    if not VIDEO_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Video model not found:\n{VIDEO_MODEL_PATH}"
        )

    return joblib.load(
        VIDEO_MODEL_PATH
    )


# ============================================================
# IMAGE FEATURE EXTRACTION
# ============================================================

def resize_gray_image(image):

    target_width = 256
    target_height = 256

    original_height, original_width = (
        image.shape[:2]
    )

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
        (
            target_height,
            target_width,
        ),
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


def extract_image_features(
    uploaded_file
):

    from src.features.image_features import (
        extract_color_features,
        resize_image,
    )

    from src.features.texture_features import (
        extract_texture_features,
    )

    from src.features.frequency_features import (
        extract_frequency_features,
    )

    file_bytes = np.asarray(
        bytearray(
            uploaded_file.getvalue()
        ),
        dtype=np.uint8,
    )

    bgr_image = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR,
    )

    if bgr_image is None:
        raise ValueError(
            "Unable to read uploaded image."
        )

    rgb_image = cv2.cvtColor(
        bgr_image,
        cv2.COLOR_BGR2RGB,
    )

    # Color
    color_image = resize_image(
        rgb_image
    )

    color_features = (
        extract_color_features(
            color_image
        )
    )

    # Grayscale
    gray_image = cv2.cvtColor(
        rgb_image,
        cv2.COLOR_RGB2GRAY,
    )

    gray_image = resize_gray_image(
        gray_image
    )

    # Texture
    texture_features = (
        extract_texture_features(
            gray_image
        )
    )

    # Frequency
    frequency_features = (
        extract_frequency_features(
            gray_image
        )
    )

    features = np.concatenate(
        [
            color_features,
            texture_features,
            frequency_features,
        ]
    )

    return features.astype(
        np.float32
    )


# ============================================================
# IMAGE PREDICTION
# ============================================================

def predict_image(
    uploaded_file
):

    model = load_image_model()

    features = extract_image_features(
        uploaded_file
    )

    expected = model.n_features_in_

    actual = len(features)

    if actual != expected:
        raise ValueError(
            f"Image model expects {expected} "
            f"features but received {actual}."
        )

    if hasattr(
        model,
        "feature_names_in_"
    ):

        X = pd.DataFrame(
            [features],
            columns=model.feature_names_in_,
        )

    else:

        X = features.reshape(
            1,
            -1
        )

    prediction = model.predict(
        X
    )[0]

    probabilities = (
        model.predict_proba(X)[0]
    )

    classes = list(
        model.classes_
    )

    probability_map = {
        int(cls): float(prob)
        for cls, prob in zip(
            classes,
            probabilities,
        )
    }

    real_probability = (
        probability_map.get(
            0,
            0.0
        )
    )

    synthetic_probability = (
        probability_map.get(
            1,
            0.0
        )
    )

    if int(prediction) == 1:

        label = "SYNTHETIC"

        confidence = (
            synthetic_probability
        )

    else:

        label = "REAL"

        confidence = (
            real_probability
        )

    return {
        "label": label,
        "confidence": confidence,
        "real_probability":
            real_probability,
        "synthetic_probability":
            synthetic_probability,
        "feature_count":
            actual,
    }


# ============================================================
# VIDEO FRAME EXTRACTION
# ============================================================

def extract_video_frames(
    video_path,
    number_of_frames=12
):

    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():
        raise ValueError(
            "Unable to open uploaded video."
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
        min(
            number_of_frames,
            total_frames
        ),
        dtype=int,
    )

    frames = []

    for index in indices:

        capture.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(index),
        )

        success, frame = (
            capture.read()
        )

        if not success:
            continue

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        frames.append(frame)

    capture.release()

    if not frames:
        raise ValueError(
            "No frames could be extracted."
        )

    return frames


# ============================================================
# MOBILE NET FEATURE EXTRACTOR
# ============================================================

@st.cache_resource
def load_mobilenet():

    weights = (
        models.MobileNet_V3_Small_Weights.DEFAULT
    )

    network = (
        models.mobilenet_v3_small(
            weights=weights
        )
    )

    extractor = torch.nn.Sequential(
        network.features,
        network.avgpool,
        torch.nn.Flatten(),
    )

    extractor.eval()

    for parameter in (
        extractor.parameters()
    ):
        parameter.requires_grad = False

    return (
        extractor,
        weights.transforms(),
    )


# ============================================================
# VIDEO FEATURE EXTRACTION
# ============================================================

def extract_video_features(
    uploaded_file
):

    # Save uploaded video temporarily
    temporary_video = (
        PROJECT_ROOT
        / "app"
        / "_uploaded_video.mp4"
    )

    with open(
        temporary_video,
        "wb"
    ) as file:

        file.write(
            uploaded_file.getbuffer()
        )

    try:

        frames = extract_video_frames(
            temporary_video,
            number_of_frames=12,
        )

        network, transform = (
            load_mobilenet()
        )

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

        with torch.no_grad():

            frame_features = network(
                batch
            )

        frame_features = (
            frame_features
            .cpu()
            .numpy()
        )

        mean_features = (
            frame_features.mean(
                axis=0
            )
        )

        std_features = (
            frame_features.std(
                axis=0
            )
        )

        max_features = (
            frame_features.max(
                axis=0
            )
        )

        min_features = (
            frame_features.min(
                axis=0
            )
        )

        video_features = np.concatenate(
            [
                mean_features,
                std_features,
                max_features,
                min_features,
            ]
        )

        return (
            video_features.astype(
                np.float32
            ),
            len(frames),
        )

    finally:

        if temporary_video.exists():
            temporary_video.unlink()


# ============================================================
# VIDEO PREDICTION
# ============================================================

def predict_video(
    uploaded_file
):

    model = load_video_model()

    features, frames_used = (
        extract_video_features(
            uploaded_file
        )
    )

    expected = model.n_features_in_

    actual = len(features)

    if actual != expected:

        raise ValueError(
            f"Video model expects {expected} "
            f"features but received {actual}."
        )

    X = features.reshape(
        1,
        -1
    )

    prediction = model.predict(
        X
    )[0]

    probabilities = (
        model.predict_proba(X)[0]
    )

    classes = list(
        model.classes_
    )

    probability_map = {
        int(cls): float(prob)
        for cls, prob in zip(
            classes,
            probabilities,
        )
    }

    real_probability = (
        probability_map.get(
            0,
            0.0
        )
    )

    synthetic_probability = (
        probability_map.get(
            1,
            0.0
        )
    )

    if int(prediction) == 1:

        label = "SYNTHETIC"

        confidence = (
            synthetic_probability
        )

    else:

        label = "REAL"

        confidence = (
            real_probability
        )

    return {
        "label": label,
        "confidence": confidence,
        "real_probability":
            real_probability,
        "synthetic_probability":
            synthetic_probability,
        "feature_count":
            actual,
        "frames_used":
            frames_used,
    }


# ============================================================
# PAGE HEADER
# ============================================================

st.title(
    "🔍 AI-Generated Media Detector"
)

st.write(
    "Detect whether an image or video "
    "is real or AI-generated."
)

st.info(
    "Image: Random Forest | "
    "Video: MobileNetV3-Small + SVM"
)


# ============================================================
# MODALITY SELECTION
# ============================================================

mode = st.radio(
    "Select media type",
    [
        "🖼️ Image",
        "🎥 Video",
    ],
    horizontal=True,
)


# ============================================================
# IMAGE UI
# ============================================================

if mode == "🖼️ Image":

    st.subheader(
        "Upload an Image"
    )

    uploaded_image = (
        st.file_uploader(
            "Choose an image",
            type=[
                "jpg",
                "jpeg",
                "png",
                "webp",
            ],
            key="image_uploader",
        )
    )

    if uploaded_image is not None:

        image = Image.open(
            uploaded_image
        )

        st.image(
            image,
            caption=uploaded_image.name,
            use_container_width=True,
        )

        if st.button(
            "🔎 Analyze Image",
            type="primary",
            use_container_width=True,
        ):

            with st.spinner(
                "Analyzing image..."
            ):

                try:

                    result = predict_image(
                        uploaded_image
                    )

                    st.divider()

                    st.subheader(
                        "Detection Result"
                    )

                    if (
                        result["label"]
                        == "REAL"
                    ):

                        st.success(
                            "✅ REAL IMAGE"
                        )

                    else:

                        st.error(
                            "⚠️ SYNTHETIC IMAGE"
                        )

                    st.metric(
                        "Confidence",
                        (
                            f"{result['confidence'] * 100:.2f}%"
                        ),
                    )

                    col1, col2 = (
                        st.columns(2)
                    )

                    with col1:

                        st.metric(
                            "Real Probability",
                            (
                                f"{result['real_probability'] * 100:.2f}%"
                            ),
                        )

                    with col2:

                        st.metric(
                            "Synthetic Probability",
                            (
                                f"{result['synthetic_probability'] * 100:.2f}%"
                            ),
                        )

                    with st.expander(
                        "Technical Details"
                    ):

                        st.write(
                            "Model: "
                            "Random Forest"
                        )

                        st.write(
                            "Features: 98"
                        )

                        st.write(
                            "Color: 60"
                        )

                        st.write(
                            "Texture: 26"
                        )

                        st.write(
                            "Frequency: 12"
                        )

                except Exception as error:

                    st.error(
                        "Image analysis failed."
                    )

                    st.exception(
                        error
                    )


# ============================================================
# VIDEO UI
# ============================================================

else:

    st.subheader(
        "Upload a Video"
    )

    uploaded_video = (
        st.file_uploader(
            "Choose a video",
            type=[
                "mp4",
                "avi",
                "mov",
                "mkv",
            ],
            key="video_uploader",
        )
    )

    if uploaded_video is not None:

        st.video(
            uploaded_video
        )

        if st.button(
            "🔎 Analyze Video",
            type="primary",
            use_container_width=True,
        ):

            with st.spinner(
                "Analyzing video... "
                "This may take some time on CPU."
            ):

                try:

                    result = predict_video(
                        uploaded_video
                    )

                    st.divider()

                    st.subheader(
                        "Detection Result"
                    )

                    if (
                        result["label"]
                        == "REAL"
                    ):

                        st.success(
                            "✅ REAL VIDEO"
                        )

                    else:

                        st.error(
                            "⚠️ SYNTHETIC VIDEO"
                        )

                    st.metric(
                        "Confidence",
                        (
                            f"{result['confidence'] * 100:.2f}%"
                        ),
                    )

                    col1, col2 = (
                        st.columns(2)
                    )

                    with col1:

                        st.metric(
                            "Real Probability",
                            (
                                f"{result['real_probability'] * 100:.2f}%"
                            ),
                        )

                    with col2:

                        st.metric(
                            "Synthetic Probability",
                            (
                                f"{result['synthetic_probability'] * 100:.2f}%"
                            ),
                        )

                    st.write(
                        "Synthetic probability"
                    )

                    st.progress(
                        float(
                            result[
                                "synthetic_probability"
                            ]
                        )
                    )

                    with st.expander(
                        "Technical Details"
                    ):

                        st.write(
                            "Model: "
                            "MobileNetV3-Small + SVM"
                        )

                        st.write(
                            "Frames sampled: "
                            f"{result['frames_used']}"
                        )

                        st.write(
                            "Features per video: "
                            f"{result['feature_count']}"
                        )

                        st.write(
                            "Device: CPU"
                        )

                except Exception as error:

                    st.error(
                        "Video analysis failed."
                    )

                    st.exception(
                        error
                    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI-Generated Media Detection Project"
)