"""Fine-tune a YOLO model on a YOLO-format road damage dataset.

Before running, prepare road_dataset/data.yaml and the train/val image + label folders.
The class names/order in data.yaml must exactly match your annotation class IDs.
"""
from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
DATA_YAML = BASE_DIR / "road_dataset" / "data.yaml"

def main():
    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"Dataset config not found: {DATA_YAML}\n"
            "Read README.md and prepare the dataset first."
        )

    # Downloads pretrained general-object weights on first run if needed.
    # Fine-tuning on your road dataset teaches the model road-damage classes.
    model = YOLO("yolo11n.pt")
    model.train(
        data=str(DATA_YAML),
        epochs=50,
        imgsz=416,
        batch=4,
        project=str(BASE_DIR / "runs" / "road_damage"),
        name="yolo11n",
        patience=10,
        plots=True
    )
    print("\nTraining finished.")
    print("Copy the run's weights/best.pt to models/best.pt.")
    print("The exact run path is printed by Ultralytics at the end of training.")

if __name__ == "__main__":
    main()
