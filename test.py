from ultralytics import YOLO

def main():
    model = YOLO("runs/detect/train/weights/best.pt")

    results = model.predict(
        source="test1.jpg",
        conf=0.25,
        save=True,
    )

    result = results[0]

    print("Detected boxes:", len(result.boxes))

    for index, box in enumerate(result.boxes, start=1):
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])
        class_name = model.names[class_id]

        print(
            f"Detection {index}: "
            f"{class_name}, confidence: {confidence:.2%}"
        )


if __name__ == "__main__":
    main()