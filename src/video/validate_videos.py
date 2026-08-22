from pathlib import Path

import cv2
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

VIDEO_ROOT = PROJECT_ROOT / "data/raw/video"

OUTPUT_DIR = PROJECT_ROOT / "data/processed/video"

METADATA_PATH = OUTPUT_DIR / "video_metadata.csv"
INVALID_PATH = OUTPUT_DIR / "invalid_videos.csv"


def get_label_and_class(video_path):
    """
    Convert Celeb-DF directory names into project labels.

    0 = real
    1 = synthetic
    """

    parent = video_path.parent.name

    if parent in ["Celeb-real", "YouTube-real"]:
        return "real", 0

    if parent == "Celeb-synthesis":
        return "synthetic", 1

    return "unknown", -1


def inspect_video(video_path):
    """
    Read video metadata using OpenCV.
    """

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        return None

    frame_count = int(
        capture.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    fps = float(
        capture.get(cv2.CAP_PROP_FPS)
    )

    width = int(
        capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    codec_value = int(
        capture.get(cv2.CAP_PROP_FOURCC)
    )

    codec = "".join(
        [
            chr((codec_value >> 0) & 0xFF),
            chr((codec_value >> 8) & 0xFF),
            chr((codec_value >> 16) & 0xFF),
            chr((codec_value >> 24) & 0xFF),
        ]
    )

    duration = (
        frame_count / fps
        if fps > 0
        else 0
    )

    # Actually read one frame to make sure
    # the video is readable, not just openable.
    success, frame = capture.read()

    capture.release()

    if not success or frame is None:
        return None

    return {
        "frame_count": frame_count,
        "fps": fps,
        "width": width,
        "height": height,
        "duration_seconds": duration,
        "codec": codec,
        "file_size_mb": (
            video_path.stat().st_size /
            (1024 * 1024)
        ),
    }


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("===== VIDEO DATASET VALIDATION =====")

    video_paths = sorted(
        VIDEO_ROOT.glob("*/**/*.mp4")
    )

    print(
        f"Videos discovered: {len(video_paths)}"
    )

    records = []
    invalid_videos = []

    for index, video_path in enumerate(
        video_paths,
        start=1,
    ):

        class_name, label = (
            get_label_and_class(video_path)
        )

        metadata = inspect_video(
            video_path
        )

        if metadata is None:

            invalid_videos.append(
                {
                    "path": str(
                        video_path.relative_to(
                            PROJECT_ROOT
                        )
                    ),
                    "class": class_name,
                    "label": label,
                }
            )

        else:

            records.append(
                {
                    "path": str(
                        video_path.relative_to(
                            PROJECT_ROOT
                        )
                    ),
                    "filename": video_path.name,
                    "source": video_path.parent.name,
                    "class": class_name,
                    "label": label,
                    **metadata,
                }
            )

        if index % 250 == 0:
            print(
                f"Processed "
                f"{index}/{len(video_paths)} videos"
            )

    df = pd.DataFrame(records)

    invalid_df = pd.DataFrame(
        invalid_videos
    )

    print(
        "\n===== VALIDATION COMPLETE ====="
    )

    print(
        f"Total videos:   {len(video_paths)}"
    )

    print(
        f"Valid videos:   {len(records)}"
    )

    print(
        f"Invalid videos: {len(invalid_videos)}"
    )

    if not df.empty:

        print(
            "\n===== CLASS COUNTS ====="
        )

        print(
            df["class"].value_counts()
        )

        print(
            "\n===== SOURCE COUNTS ====="
        )

        print(
            df["source"].value_counts()
        )

        print(
            "\n===== RESOLUTION SUMMARY ====="
        )

        print(
            df[
                ["width", "height"]
            ].describe()
        )

        print(
            "\n===== FPS SUMMARY ====="
        )

        print(
            df["fps"].describe()
        )

        print(
            "\n===== DURATION SUMMARY (SECONDS) ====="
        )

        print(
            df["duration_seconds"].describe()
        )

        print(
            "\n===== FILE SIZE SUMMARY (MB) ====="
        )

        print(
            df["file_size_mb"].describe()
        )

    df.to_csv(
        METADATA_PATH,
        index=False,
    )

    invalid_df.to_csv(
        INVALID_PATH,
        index=False,
    )

    print(
        f"\nMetadata saved to: "
        f"{METADATA_PATH}"
    )

    print(
        f"Invalid videos saved to: "
        f"{INVALID_PATH}"
    )


if __name__ == "__main__":
    main()