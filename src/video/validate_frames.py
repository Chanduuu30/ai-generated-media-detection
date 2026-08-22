from pathlib import Path

import cv2
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FRAME_ROOT = (
    PROJECT_ROOT /
    "data/processed/video/frames"
)

OUTPUT_PATH = (
    PROJECT_ROOT /
    "data/processed/video/frame_metadata.csv"
)

INVALID_PATH = (
    PROJECT_ROOT /
    "data/processed/video/invalid_frames.csv"
)


def inspect_frame(frame_path):

    image = cv2.imread(
        str(frame_path)
    )

    if image is None:
        return None

    height, width = image.shape[:2]

    if height <= 0 or width <= 0:
        return None

    return {
        "width": width,
        "height": height,
        "file_size_kb": (
            frame_path.stat().st_size /
            1024
        ),
    }


def main():

    print("===== FRAME VALIDATION =====")

    frame_paths = sorted(
        FRAME_ROOT.glob(
            "*/*/*/*.jpg"
        )
    )

    print(
        f"Frames discovered: "
        f"{len(frame_paths)}"
    )

    records = []
    invalid_frames = []

    for index, frame_path in enumerate(
        frame_paths,
        start=1,
    ):

        # Expected structure:
        # frames/split/class/video/frame.jpg

        relative = frame_path.relative_to(
            FRAME_ROOT
        )

        parts = relative.parts

        if len(parts) != 4:
            invalid_frames.append(
                {
                    "path": str(
                        frame_path.relative_to(
                            PROJECT_ROOT
                        )
                    ),
                    "reason": "unexpected_path_structure",
                }
            )
            continue

        split = parts[0]
        class_name = parts[1]
        video_id = parts[2]

        metadata = inspect_frame(
            frame_path
        )

        if metadata is None:

            invalid_frames.append(
                {
                    "path": str(
                        frame_path.relative_to(
                            PROJECT_ROOT
                        )
                    ),
                    "split": split,
                    "class": class_name,
                    "video_id": video_id,
                    "reason": "cannot_read_image",
                }
            )

        else:

            records.append(
                {
                    "path": str(
                        frame_path.relative_to(
                            PROJECT_ROOT
                        )
                    ),
                    "split": split,
                    "class": class_name,
                    "video_id": video_id,
                    "filename": frame_path.name,
                    **metadata,
                }
            )

        if index % 2000 == 0:

            print(
                f"Processed "
                f"{index}/{len(frame_paths)} frames"
            )

    df = pd.DataFrame(
        records
    )

    invalid_df = pd.DataFrame(
        invalid_frames
    )

    print(
        "\n===== VALIDATION COMPLETE ====="
    )

    print(
        f"Total frames:   {len(frame_paths)}"
    )

    print(
        f"Valid frames:   {len(records)}"
    )

    print(
        f"Invalid frames: {len(invalid_frames)}"
    )

    if not df.empty:

        print(
            "\n===== SPLIT COUNTS ====="
        )

        print(
            df["split"].value_counts()
        )

        print(
            "\n===== CLASS COUNTS ====="
        )

        print(
            df["class"].value_counts()
        )

        print(
            "\n===== SPLIT × CLASS ====="
        )

        print(
            pd.crosstab(
                df["split"],
                df["class"],
            )
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
            "\n===== FILE SIZE SUMMARY (KB) ====="
        )

        print(
            df["file_size_kb"].describe()
        )

        print(
            "\n===== FRAMES PER VIDEO ====="
        )

        frames_per_video = (
            df.groupby(
                [
                    "split",
                    "class",
                    "video_id",
                ]
            )
            .size()
        )

        print(
            frames_per_video.describe()
        )

        print(
            "\n===== VIDEOS REPRESENTED ====="
        )

        print(
            df.groupby(
                [
                    "split",
                    "class",
                ]
            )["video_id"]
            .nunique()
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    invalid_df.to_csv(
        INVALID_PATH,
        index=False,
    )

    print(
        f"\nFrame metadata saved to: "
        f"{OUTPUT_PATH}"
    )

    print(
        f"Invalid frame report saved to: "
        f"{INVALID_PATH}"
    )


if __name__ == "__main__":
    main()