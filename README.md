# Multimodal Fake Content Detection Using Machine Learning

A machine-learning system that detects whether digital images and videos are real or AI-generated/synthetic. The project implements two independent detection pipelines—one for images using handcrafted features and a Random Forest classifier, and one for videos using MobileNetV3-Small deep features and a Linear SVM—along with a validation experiment for weighted multimodal fusion.

---

## Overview

This repository contains implementations of image and video fake-content detection pipelines, together with a validation experiment for weighted multimodal fusion. The system addresses the growing challenge of distinguishing authentic media from AI-generated and manipulated content.

**Implemented modalities:**
- **Image detection**: 98 handcrafted features (color, texture, frequency) → Random Forest
- **Video detection**: 12-frame sampling → MobileNetV3-Small (576-d/frame) → statistical aggregation (2,304-d) → Linear SVM
- **Multimodal validation**: Weighted probability fusion of image and video outputs (validation results only)

**Not implemented:** Text detection, audio detection, or end-to-end joint training.

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Image pipeline** | Color (60) + Texture/LBP (26) + Frequency/FFT (12) = 98 features → Random Forest (300 trees) |
| **Video pipeline** | Evenly spaced 12-frame sampling → MobileNetV3-Small (ImageNet pretrained) → Mean/Std/Max/Min aggregation → Linear SVM (C=1.0) |
| **Multimodal fusion** | Weighted probability combination evaluated on validation set |
| **Streamlit UI** | Web interface for uploading and analyzing images/videos |
| **CLI inference** | `python -m src.inference.predict --image/--video <path>` |
| **Data validation** | Automated duplicate, corruption, and path-leakage checks |

---

## System Architecture

The system implements two independent pipelines:

**Image Pipeline:**
```
Input Image → Resize 256×256 (pad) → Color (60) + Texture/LBP (26) + Frequency/FFT (12) → 98 Features → Random Forest → Prediction
```

**Video Pipeline:**
```
Input Video → 12-Frame Sampling → MobileNetV3-Small (576/frame) → Mean/Std/Max/Min Aggregation → 2,304 Features → Linear SVM → Prediction
```

Both pipelines output class label (REAL/SYNTHETIC), confidence, and class probabilities via CLI or Streamlit UI.

---

## Dataset

### Image Data (MediaEval ITW-SM)

| Path | Classes | Count |
|------|---------|-------|
| `data/raw/mediaeval/ITW-SM/0_real/` | Real (0) | [Not specified in repository] |
| `data/raw/mediaeval/ITW-SM/1_fake/` | Synthetic (1) | [Not specified in repository] |

**Preprocessing:** Aspect-ratio-preserving resize to 256×256 with center padding, RGB normalization to [0,1], HSV conversion, grayscale for texture/frequency.

### Video Data (Celeb-DF-v2)

| Directory | Class | Videos |
|-----------|-------|--------|
| `data/raw/video/Celeb-real/` | Real (0) | 890 |
| `data/raw/video/Celeb-synthesis/` | Synthetic (1) | 5,639 |
| **Total** | | **6,529** |

**Verified split:**

| Split | Real | Synthetic | Total |
|-------|------|-----------|-------|
| Train | 586 | 4,364 | 4,950 |
| Validation | 126 | 935 | 1,061 |
| Test | 178 | 340 | 518 |

**Frames extracted:** 78,337 total (Train: 59,389, Validation: 12,732, Test: 6,216)

> The video dataset is Celeb-DF-v2 (identified from `data/raw/video/Celeb-DF-v2.zip` and directory structure).

---

## Image Detection Pipeline

### Preprocessing
- Load image with OpenCV (BGR → RGB)
- Aspect-ratio-preserving resize to 256×256 with center padding
- RGB normalization to [0, 1] for color features
- Grayscale conversion for texture and frequency features

### Feature Extraction (98 total)

| Group | Count | Method | Verified Parameters |
|-------|-------|--------|---------------------|
| **Color** | 60 | RGB/HSV statistics + RGB histograms | RGB mean/std (6), HSV mean/std (6), RGB 16-bin histograms (48) |
| **Texture** | 26 | Local Binary Pattern (LBP) | `P=24`, `R=3`, `method="uniform"` → 26-bin histogram |
| **Frequency** | 12 | 2D Fourier Transform | FFT → fftshift → log1p(magnitude) → statistical descriptors + low/high frequency energy ratio |

### Classifier: Random Forest

Saved model: `models/final_random_forest.pkl`

**Verified hyperparameters:**
```python
{
    'bootstrap': True,
    'ccp_alpha': 0.0,
    'class_weight': None,
    'criterion': 'gini',
    'max_depth': None,
    'max_features': 'sqrt',
    'max_leaf_nodes': None,
    'max_samples': None,
    'min_impurity_decrease': 0.0,
    'min_samples_leaf': 1,
    'min_samples_split': 2,
    'min_weight_fraction_leaf': 0.0,
    'monotonic_cst': None,
    'n_estimators': 300,
    'n_jobs': -1,
    'oob_score': False,
    'random_state': 42,
    'verbose': 0,
    'warm_start': False
}
```

---

## Video Detection Pipeline

### Frame Processing
- Evenly spaced frame sampling: 12 frames per video (or all frames if fewer than 12); first and last frames excluded during dataset construction, included during inference
- BGR → RGB conversion
- TorchVision default MobileNetV3-Small preprocessing

### Feature Extraction
- **Backbone:** MobileNetV3-Small (ImageNet pretrained, frozen)
- **Features per frame:** 576 (output of `features` + `avgpool` + `Flatten`)
- **Aggregation:** Mean, Std, Max, Min across 12 frames → 4 × 576 = 2,304 features/video

### Classifier: Linear SVM

Saved model: `models/video/video_all_frame_svm.pkl`

**Training configuration:**
```python
{
    'C': 1.0,
    'kernel': 'linear',
    'class_weight': 'balanced',
    'probability': True,
    'random_state': 42
}
```

**Note:** The SVM uses `probability=True` which enables `predict_proba` via Platt scaling. A deprecation warning may appear during inference.

---

## Multimodal Validation

A validation experiment combining image and video model probabilities using weighted fusion. **These are validation results only—no official multimodal test evaluation exists in the repository.**

| Image Weight | Video Weight | Accuracy | Precision | Recall | F1 | ROC-AUC |
|--------------|--------------|----------|-----------|--------|-----|---------|
| 0.25 | 0.75 | 59.52% | 55.26% | 100.00% | 71.19% | 85.85% |
| **0.50** | **0.50** | **66.67%** | **60.00%** | **100.00%** | **75.00%** | **89.92%** |
| 0.75 | 0.25 | 77.38% | 71.43% | 91.27% | 80.14% | 89.14% |

**Highest validation ROC-AUC among evaluated fusion weights:** 89.92% (image 0.50 / video 0.50)  
**Highest validation accuracy among evaluated fusion weights:** 77.38% (image 0.75 / video 0.25)

⚠️ **Important:** These results are from the validation set only. The fusion weights have not been evaluated on a held-out multimodal test set.

---

## Model Configuration

### Random Forest (Image)
See [Image Detection Pipeline](#classifier-random-forest) for complete verified parameters.

### Linear SVM (Video)
See [Video Detection Pipeline](#classifier-linear-svm) for complete verified parameters.

---

## Evaluation Methodology

- **Image evaluation:** Uses the saved test results in `models/final_test_results.csv`, generated by evaluating the trained Random Forest on a held-out image test split.
- **Video evaluation:** Uses the saved test results in `models/video/final_video_test_results.csv`, generated by evaluating the trained Linear SVM on the held-out video test split (518 videos).
- **Multimodal fusion evaluation:** Uses the validation results in `models/multimodal_validation_results.csv`, generated by combining image and video model probabilities on the validation splits only. No held-out multimodal test evaluation is currently available.

---

## Results

### Held-Out Test Set Performance

| Modality | Model | Test Samples | Accuracy | Precision | Recall | F1 | ROC-AUC |
|----------|-------|--------------|----------|-----------|--------|-----|---------|
| **Image** | Random Forest (98 features) | [Not specified in repository] | **80.83%** | **82.11%** | **78.93%** | **80.49%** | **89.14%** |
| **Video** | MobileNetV3-Small + Linear SVM | 518 (178 real, 340 synthetic) | **72.78%** | **76.12%** | **85.29%** | **80.44%** | **67.06%** |

**Sources:** `models/final_test_results.csv` (image), `models/video/final_video_test_results.csv` (video)

> The video results correspond to the **held-out video test set** (518 videos never seen during training/validation). The image test set size is not recorded in the repository.

### Video Confusion Matrix (Held-Out Test)

```
                 Predicted
                 Real  Synthetic
Actual Real        87        91
Actual Synthetic   50       290
```

**Per-class metrics (video test):**

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| Real | 0.64 | 0.49 | 0.55 | 178 |
| Synthetic | 0.76 | 0.85 | 0.80 | 340 |

---

## Sample Predictions

Verified inference outputs from the CLI tool:

| Input | Type | Prediction | Confidence | Real Prob. | Synthetic Prob. | Model |
|-------|------|------------|------------|------------|-----------------|-------|
| `data/raw/mediaeval/ITW-SM/0_real/X_real_2958.jpg` | Real image | REAL | 83.67% | 83.67% | 16.33% | Random Forest |
| `data/raw/mediaeval/ITW-SM/1_fake/x_807.jpg` | Synthetic image | SYNTHETIC | 72.67% | 27.33% | 72.67% | Random Forest |
| `data/raw/video/Celeb-real/id48_0009.mp4` | Real video | SYNTHETIC | 83.84% | 16.16% | 83.84% | MobileNetV3-Small + SVM |
| `data/raw/video/Celeb-synthesis/id26_id9_0002.mp4` | Synthetic video | SYNTHETIC | 96.56% | 3.44% | 96.56% | MobileNetV3-Small + SVM |

> Note: The real video from `Celeb-real` was incorrectly classified as synthetic, illustrating a known limitation.

---

## Project Structure

```
ai-generated-media-detection/
├── app/
│   └── streamlit_app.py          # Streamlit web interface
├── data/
│   ├── processed/                # Extracted features (gitignored)
│   │   ├── all_image_features.csv
│   │   ├── color_texture_features.csv
│   │   ├── texture_features.csv
│   │   ├── frequency_features.csv
│   │   ├── image_features.csv
│   │   ├── image_metadata.csv
│   │   └── video/
│   │       ├── video_all_frame_features.csv
│   │       ├── video_cnn_features.csv
│   │       ├── video_metadata.csv
│   │       ├── frame_metadata.csv
│   │       ├── invalid_videos.csv
│   │       ├── invalid_frames.csv
│   │       ├── video_duplicates.csv
│   │       └── frames/                 # Extracted frames (gitignored)
│   ├── raw/                      # Datasets (gitignored)
│   │   ├── mediaeval/ITW-SM/0_real/
│   │   ├── mediaeval/ITW-SM/1_fake/
│   │   └── video/
│   │       ├── Celeb-real/
│   │       ├── Celeb-synthesis/
│   │       ├── YouTube-real/
│   │       └── Celeb-DF-v2.zip
│   └── splits/
│       ├── image_splits.csv
│       └── video/video_splits.csv
├── models/
│   ├── final_random_forest.pkl          # Image model
│   ├── final_test_results.csv           # Image test metrics
│   ├── multimodal_validation_model.pkl
│   ├── multimodal_validation_results.csv
│   └── video/
│       ├── video_all_frame_svm.pkl              # Final selected video model
│       ├── video_all_frame_svm_results.csv
│       ├── final_video_test_results.csv         # Held-out video test metrics
│       ├── video_all_frame_logistic.pkl
│       ├── video_all_frame_logistic_results.csv
│       ├── video_random_forest.pkl
│       ├── video_random_forest_results.csv
│       ├── video_cpu_baseline.pkl
│       └── video_cpu_baseline_results.csv
├── src/
│   ├── data/
│   │   ├── create_splits.py
│   │   └── validate_images.py
│   ├── features/
│   │   ├── image_features.py          # Color features
│   │   ├── texture_features.py        # LBP
│   │   ├── frequency_features.py      # FFT
│   │   ├── combine_features.py
│   │   ├── combine_all_features.py
│   │   ├── build_image_features.py
│   │   ├── build_texture_features.py
│   │   ├── build_frequency_features.py
│   │   └── verify_features.py
│   ├── inference/
│   │   └── predict.py                 # CLI inference
│   ├── models/
│   │   ├── train_all_features_random_forest.py
│   │   ├── train_color_texture_random_forest.py
│   │   ├── train_frequency_random_forest.py
│   │   ├── train_logistic_regression.py
│   │   ├── train_multimodal_validation.py
│   │   ├── train_random_forest.py
│   │   ├── train_svm.py
│   │   ├── train_texture_logistic.py
│   │   ├── train_color_texture_logistic.py
│   │   └── final_test_evaluation.py
│   ├── video/
│   │   ├── extract_frames.py
│   │   ├── validate_videos.py
│   │   ├── validate_frames.py
│   │   ├── build_all_frame_video_features.py
│   │   ├── train_video_cpu_baseline.py
│   │   ├── train_video_random_forest.py
│   │   ├── train_all_frame_logistic.py
│   │   ├── train_all_frame_svm.py
│   │   ├── train_video_cnn.py
│   │   ├── create_video_splits.py
│   │   ├── check_duplicates.py
│   │   └── final_video_test_evaluation.py
│   └── eda/
│       ├── check_duplicates.py
│       ├── classwise_analysis.py
│       ├── image_metadata.py
│       └── visual_samples.py
├── external/
│   └── CNNDetection/                # External reference implementation
├── experiments/                     # Experiment scripts
├── notebooks/                       # Jupyter notebooks
├── reports/                         # Generated reports
├── .gitignore
├── run.sh                           # Launch script for Streamlit
└── README.md
```

---

## Installation

The repository includes a `requirements.txt` file that is currently empty. Dependencies can be installed manually using the commands below (verified from source code imports):

```bash
# Core
pip install numpy pandas scikit-learn joblib

# Image processing
pip install opencv-python pillow scikit-image

# Deep learning
pip install torch torchvision

# Web UI
pip install streamlit
```

**Python version:** Tested on Python 3.12 (venv at `.venv/`)

---

## Usage

### Streamlit Web Interface

```bash
# From project root
python -m streamlit run app/streamlit_app.py
# Open http://localhost:8501
```

Or use the provided launch script:
```bash
./run.sh
```

**UI features:**
- Image tab: Upload JPG/PNG/WebP → preview → analyze → shows label, confidence, probabilities, feature breakdown
- Video tab: Upload MP4/AVI/MOV/MKV → preview player → analyze → shows label, confidence, probabilities, frames used, progress bar

### CLI Inference

**Image:**
```bash
python -m src.inference.predict --image path/to/image.jpg
```

**Video:**
```bash
python -m src.inference.predict --video path/to/video.mp4
```

### Example CLI Output (Image)

```
========================================
        IMAGE AI-DETECTION
========================================
Input: /path/to/image.jpg

===== EXTRACTING IMAGE FEATURES =====
Color features: 60
Texture features: 26
Frequency features: 12
Combined feature count: 98

Model expects: 98 features
Features supplied: 98

===== RESULT =====
Prediction: REAL
Confidence: 83.67%
Real probability: 83.67%
Synthetic probability: 16.33%
Model: Random Forest
```

### Example CLI Output (Video)

```
========================================
        VIDEO AI-DETECTION
========================================
Input: /path/to/video.mp4

Frames extracted: 12
Frame feature shape: (12, 576)
Video feature count: 2304

Model expects: 2304 features
Features supplied: 2304

===== RESULT =====
Prediction: SYNTHETIC
Confidence: 96.56%
Real probability: 3.44%
Synthetic probability: 96.56%
Model: MobileNetV3-Small + Linear SVM
```

---

## Output Format

Both interfaces return:

| Field | Description |
|-------|-------------|
| `Prediction` | `REAL` (class 0) or `SYNTHETIC` (class 1) |
| `Confidence` | Probability of the predicted class |
| `Real probability` | P(class=0) |
| `Synthetic probability` | P(class=1) |
| `Model` | `Random Forest` (image) or `MobileNetV3-Small + Linear SVM` (video) |
| `Feature count` | 98 (image) or 2,304 (video) |
| `Frames used` | Video only (typically 12) |

---

## Saved Models and Results

| File | Description |
|------|-------------|
| `models/final_random_forest.pkl` | Trained image Random Forest |
| `models/final_test_results.csv` | Image test metrics (accuracy, precision, recall, F1, ROC-AUC) |
| `models/video/video_all_frame_svm.pkl` | Final selected video model |
| `models/video/final_video_test_results.csv` | Held-out video test metrics |
| `models/multimodal_validation_results.csv` | Fusion validation results (3 weight combinations) |
| `models/multimodal_validation_model.pkl` | Fusion model artifact |

---

## Limitations

| Limitation | Evidence |
|------------|----------|
| **Video performance lower than image** | Video test accuracy 72.78% vs image 80.83%; video ROC-AUC 67.06% vs image 89.14% |
| **Class imbalance in video data** | 6:1 synthetic-to-real ratio (5,639 vs 890) |
| **Real video misclassified** | Sample `Celeb-real/id48_0009.mp4` predicted as synthetic with 83.84% confidence |
| **No multimodal test evaluation** | Only validation fusion results available |
| **Fixed 12-frame sampling** | Short videos duplicate frames; long videos undersampled |
| **Linear SVM kernel** | The final video classifier uses a linear SVM kernel; nonlinear classifier alternatives were explored separately |
| **SVM probability calibration** | Platt scaling (`probability=True`) with deprecation warning observed |
| **Random Forest feature-name warning** | During CLI inference, scikit-learn reports that the input does not contain the feature names used during model fitting. The warning does not prevent prediction from completing. Streamlit constructs a DataFrame with matching feature names, avoiding the warning. |
| **Single video dataset** | Generalization to other synthesis types and datasets untested |

---

## Future Improvements

*Proposed work—not yet implemented.*

- **Temporal modeling**: LSTM/Transformer on frame sequences instead of statistical aggregation
- **Improved multimodal fusion**: Learnable fusion, attention-based weighting, stacking
- **Class imbalance handling**: Focal loss, resampling, threshold optimization
- **Probability calibration**: Temperature scaling, isotonic regression for both models
- **Cross-dataset evaluation**: FaceForensics++, DFDC, WildDeepfake, DiffusionDB
- **Adversarial robustness**: Testing against perturbations, compression, resizing
- **Broader synthetic-media coverage**: Evaluate the system on additional generation and manipulation techniques
- **Explainability**: SHAP for Random Forest, Grad-CAM/attention for video
- **Real-time streaming**: Frame-level inference with temporal smoothing
- **GPU acceleration**: Batch inference, mixed precision for video
- **Official multimodal test set**: Curate paired image/video test data for fusion evaluation

---

## Technical Stack

| Library | Purpose |
|---------|---------|
| Python 3.12 | Core language |
| PyTorch + torchvision | MobileNetV3-Small feature extraction |
| scikit-learn | Random Forest, Linear SVM, metrics, preprocessing |
| OpenCV (cv2) | Image/video I/O, color space conversion, resizing |
| NumPy | Numerical computation, FFT, array operations |
| pandas | DataFrame handling, CSV I/O |
| scikit-image | Local Binary Pattern (LBP) |
| joblib | Model serialization |
| Streamlit | Web UI |
| Pillow (PIL) | Image loading for torchvision transforms |

---

## Reproducibility

1. **Clone and set up environment:**
   ```bash
   git clone <repo-url>
   cd ai-generated-media-detection
   python -m venv .venv
   source .venv/bin/activate
   pip install numpy pandas scikit-learn joblib opencv-python pillow scikit-image torch torchvision streamlit
   ```

2. **Verify models exist:**
   ```bash
   ls models/final_random_forest.pkl models/video/video_all_frame_svm.pkl
   ```

3. **Run inference on sample data:**
   ```bash
   python -m src.inference.predict --image data/raw/mediaeval/ITW-SM/0_real/X_real_2958.jpg
   python -m src.inference.predict --video data/raw/video/Celeb-synthesis/id26_id9_0002.mp4
   ```

4. **Launch web UI:**
   ```bash
   python -m streamlit run app/streamlit_app.py
   ```

Training scripts are available in `src/models/` and `src/video/` but require the raw datasets (not included in the repository).

---

## Project Status

**Current state:** Research prototype with implemented image and video detection pipelines, a validation-only multimodal fusion experiment, and functional CLI and Streamlit interfaces.

- ✅ Image detection: Implemented and evaluated on held-out test
- ✅ Video detection: Implemented and evaluated on held-out video test set
- ✅ Multimodal fusion: Validation experiment completed (no test evaluation)
- ✅ Interfaces: Streamlit UI and CLI functional

This is an academic/project submission, not a production-ready system.

---

## License

This project is for research and educational purposes. No license file is included in the repository.

---

*YAVA Internship Submission — Multimodal Fake Content Detection Using Deep Learning*