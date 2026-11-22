
from ultralytics import YOLO

if __name__ == "__main__":

    # Load a small pretrained YOLO model
    model = YOLO("yolo11n.pt")

    # Train using the Indian road damage dataset
    model.train(
        data="road_dataset_india/data.yaml",
        epochs=30,
        imgsz=416,
        batch=2,
        device="cpu",
        workers=2,
        patience=8,
        cache=False,
        name="india_road_damage"
    )

    print("Training completed!")
