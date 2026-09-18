#!/bin/bash

set -e

echo "=============================================="
echo "   MULTIMODAL AI-GENERATED MEDIA DETECTION"
echo "=============================================="

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

echo
echo "Checking project files..."

if [ ! -f "models/final_random_forest.pkl" ]; then
    echo "ERROR: Image model not found."
    exit 1
fi

if [ ! -f "models/video/video_all_frame_svm.pkl" ]; then
    echo "ERROR: Video model not found."
    exit 1
fi

if [ ! -f "app/streamlit_app.py" ]; then
    echo "ERROR: Streamlit application not found."
    exit 1
fi

echo "✓ Image model found"
echo "✓ Video model found"
echo "✓ Streamlit application found"

echo
echo "=============================================="
echo "Starting Streamlit application..."
echo "=============================================="
echo
echo "Open: http://localhost:8501"
echo

python -m streamlit run app/streamlit_app.py