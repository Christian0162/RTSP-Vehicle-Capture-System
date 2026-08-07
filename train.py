from ultralytics import YOLO

def main():
    #actual training
    model = YOLO("yolo11n.pt")
    model.train(
        data=r"C:\Users\Big Blue Logistics\Desktop\New folder\data.yaml",
        epochs=100,
        imgsz=640,
    )

    #resuming the training
    # model = YOLO("runs/detect/train/weights/last.pt")
    # model.train(resume=True)


if __name__ == "__main__":
    main()