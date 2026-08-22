from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FRAME_METADATA = (
    PROJECT_ROOT /
    "data/processed/video/frame_metadata.csv"
)

MODEL_DIR = PROJECT_ROOT / "models/video"

MODEL_PATH = MODEL_DIR / "video_cnn_baseline.pth"

RESULTS_PATH = (
    PROJECT_ROOT /
    "models/video/video_cnn_training_results.csv"
)

IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 1e-4
NUM_WORKERS = 2
RANDOM_SEED = 42


class FrameDataset(Dataset):

    def __init__(self, dataframe, transform=None):

        self.df = dataframe.reset_index(
            drop=True
        )

        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        image_path = (
            PROJECT_ROOT /
            row["path"]
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(
            int(row["class"] == "synthetic"),
            dtype=torch.long,
        )

        return image, label


def create_model():

    weights = (
        models.ResNet18_Weights.DEFAULT
    )

    model = models.resnet18(
        weights=weights
    )

    num_features = (
        model.fc.in_features
    )

    model.fc = nn.Linear(
        num_features,
        2,
    )

    return model


def evaluate(model, loader, device):

    model.eval()

    correct = 0
    total = 0

    loss_sum = 0.0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels,
            )

            loss_sum += (
                loss.item() *
                labels.size(0)
            )

            predictions = (
                outputs.argmax(dim=1)
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    accuracy = (
        correct / total
        if total > 0
        else 0
    )

    average_loss = (
        loss_sum / total
        if total > 0
        else 0
    )

    return average_loss, accuracy


def main():

    torch.manual_seed(
        RANDOM_SEED
    )

    np.random.seed(
        RANDOM_SEED
    )

    print(
        "===== VIDEO CNN BASELINE ====="
    )

    print(
        f"PyTorch version: "
        f"{torch.__version__}"
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    if device.type == "cuda":

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    print(
        "\nLoading frame metadata..."
    )

    df = pd.read_csv(
        FRAME_METADATA
    )

    train_df = df[
        df["split"] == "train"
    ].copy()

    validation_df = df[
        df["split"] == "validation"
    ].copy()

    print(
        f"Training frames: "
        f"{len(train_df)}"
    )

    print(
        f"Validation frames: "
        f"{len(validation_df)}"
    )

    print(
        "\n===== FRAME CLASS DISTRIBUTION ====="
    )

    print(
        train_df["class"].value_counts()
    )

    # ImageNet normalization.
    normalize = transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406,
        ],
        std=[
            0.229,
            0.224,
            0.225,
        ],
    )

    train_transform = transforms.Compose(
        [
            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE)
            ),
            transforms.RandomHorizontalFlip(
                p=0.5
            ),
            transforms.RandomRotation(
                degrees=5
            ),
            transforms.ToTensor(),
            normalize,
        ]
    )

    validation_transform = transforms.Compose(
        [
            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE)
            ),
            transforms.ToTensor(),
            normalize,
        ]
    )

    train_dataset = FrameDataset(
        train_df,
        transform=train_transform,
    )

    validation_dataset = FrameDataset(
        validation_df,
        transform=validation_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    print(
        "\n===== CREATING MODEL ====="
    )

    model = create_model()

    model = model.to(device)

    # Handle the strong class imbalance.
    class_counts = (
        train_df["class"]
        .value_counts()
    )

    real_count = class_counts.get(
        "real",
        1,
    )

    synthetic_count = class_counts.get(
        "synthetic",
        1,
    )

    total = (
        real_count +
        synthetic_count
    )

    class_weights = torch.tensor(
        [
            total / (2 * real_count),
            total / (2 * synthetic_count),
        ],
        dtype=torch.float32,
        device=device,
    )

    print(
        "\nClass weights:"
    )

    print(
        class_weights
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    print(
        "\n===== TRAINING ====="
    )

    history = []

    best_validation_accuracy = 0.0

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        model.train()

        running_loss = 0.0
        correct = 0
        total_samples = 0

        for batch_index, (
            images,
            labels,
        ) in enumerate(train_loader):

            images = images.to(
                device,
                non_blocking=True,
            )

            labels = labels.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad()

            outputs = model(
                images
            )

            loss = criterion(
                outputs,
                labels,
            )

            loss.backward()

            optimizer.step()

            running_loss += (
                loss.item() *
                labels.size(0)
            )

            predictions = (
                outputs.argmax(dim=1)
            )

            correct += (
                predictions == labels
            ).sum().item()

            total_samples += (
                labels.size(0)
            )

        train_loss = (
            running_loss /
            total_samples
        )

        train_accuracy = (
            correct /
            total_samples
        )

        validation_loss, validation_accuracy = (
            evaluate(
                model,
                validation_loader,
                device,
            )
        )

        print(
            f"\nEpoch {epoch}/{EPOCHS}"
        )

        print(
            f"Train loss: "
            f"{train_loss:.4f}"
        )

        print(
            f"Train accuracy: "
            f"{train_accuracy:.4f}"
        )

        print(
            f"Validation loss: "
            f"{validation_loss:.4f}"
        )

        print(
            f"Validation accuracy: "
            f"{validation_accuracy:.4f}"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
            }
        )

        if (
            validation_accuracy >
            best_validation_accuracy
        ):

            best_validation_accuracy = (
                validation_accuracy
            )

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),
                    "validation_accuracy":
                        validation_accuracy,
                    "epoch": epoch,
                },
                MODEL_PATH,
            )

            print(
                "Best model checkpoint saved."
            )

    history_df = pd.DataFrame(
        history
    )

    history_df.to_csv(
        RESULTS_PATH,
        index=False,
    )

    print(
        "\n===== TRAINING COMPLETE ====="
    )

    print(
        f"Best validation accuracy: "
        f"{best_validation_accuracy:.4f}"
    )

    print(
        f"Model saved to: "
        f"{MODEL_PATH}"
    )

    print(
        f"Training results saved to: "
        f"{RESULTS_PATH}"
    )


if __name__ == "__main__":
    main()