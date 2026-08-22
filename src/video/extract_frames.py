from pathlib import Path

import cv2
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SPLIT_PATH = (
    PROJECT_ROOT /
    "data/splits/video/video_splits.csv"
)

OUTPUT_ROOT = (
    PROJECT_ROOT /
    "data/processed/video/frames"
)

FRAMES_PER_VIDEO = 12


def extract_video_frames(video_path, output_dir):
    """
    Extract evenly spaced frames from a video.
    """

    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():
        return 0

    frame_count = int(
        capture.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if frame_count <= 0:
        capture.release()
        return 0

    # Avoid selecting the first/last frame.
    frame_indices = np.linspace(
        0,
        frame_count - 1,
        FRAMES_PER_VIDEO + 2,
        dtype=int,
    )[1:-1]

    saved = 0

    for frame_index in frame_indices:

        capture.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(frame_index),
        )

        success, frame = capture.read()

        if not success or frame is None:
            continue

        frame_name = (
            f"frame_{saved:02d}.jpg"
        )

        output_path = (
            output_dir /
            frame_name
        )

        success = cv2.imwrite(
            str(output_path),
            frame,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                95,
            ],
        )

        if success:
            saved += 1

    capture.release()

    return saved


def main():

    print("===== VIDEO FRAME EXTRACTION =====")

    df = pd.read_csv(
        SPLIT_PATH
    )

    print(
        f"Videos to process: {len(df)}"
    )

    print(
        f"Frames per video: "
        f"{FRAMES_PER_VIDEO}"
    )

    total_frames = 0
    failed_videos = []

    for index, row in df.iterrows():

        video_path = (
            PROJECT_ROOT /
            row["path"]
        )

        split = row["split"]
        class_name = row["class"]

        video_stem = (
            Path(row["path"]).stem
        )

        output_dir = (
            OUTPUT_ROOT /
            split /
            class_name /
            video_stem
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        saved = extract_video_frames(
            video_path,
            output_dir,
        )

        if saved == 0:

            failed_videos.append(
                {
                    "path": row["path"],
                    "split": split,
                    "class": class_name,
                }
            )

        total_frames += saved

        if (index + 1) % 250 == 0:

            print(
                f"Processed "
                f"{index + 1}/{len(df)} videos "
                f"| Frames saved: {total_frames}"
            )

    print(
        "\n===== EXTRACTION COMPLETE ====="
    )

    print(
        f"Videos processed: {len(df)}"
    )

    print(
        f"Total frames saved: {total_frames}"
    )

    print(
        f"Failed videos: {len(failed_videos)}"
    )

    if failed_videos:

        failed_path = (
            PROJECT_ROOT /
            "data/processed/video/"
            "failed_frame_extraction.csv"
        )

        pd.DataFrame(
            failed_videos
        ).to_csv(
            failed_path,
            index=False,
        )

        print(
            f"Failed video list saved to: "
            f"{failed_path}"
        )

    # Count extracted frames by split/class.
    print(
        "\n===== FRAME COUNTS ====="
    )

    for split in [
        "train",
        "validation",
        "test",
    ]:

        for class_name in [
            "real",
            "synthetic",
        ]:

            frame_dir = (
                OUTPUT_ROOT /
                split /
                class_name
            )

            if frame_dir.exists():

                count = len(
                    list(
                        frame_dir.glob(
                            "*/*.jpg"
                        )
                    )
                )

            else:
                count = 0

            print(
                f"{split:12s} "
                f"{class_name:10s}: "
                f"{count}"
            )


if __name__ == "__main__":
    main()