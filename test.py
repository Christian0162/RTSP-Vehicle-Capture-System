from ultralytics import YOLO


def main():
    model = YOLO("runs/detect/train-2/weights/best.pt")

    results = model.predict(
        source="test1.png",
        conf=0.10,
        save=True,
    )

    result = results[0]

    print("Available classes:")
    for class_id, class_name in model.names.items():
        print(f"{class_id}: {class_name}")

    print()
    print("Detected objects:", len(result.boxes))

    for index, box in enumerate(result.boxes, start=1):
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])
        class_name = model.names[class_id]

        print(
            f"Detection {index}: "
            f"class_id={class_id}, "
            f"class={class_name}, "
            f"confidence={confidence:.2%}"
        )


if __name__ == "__main__":
    main()