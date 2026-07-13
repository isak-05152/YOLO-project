from ultralytics import YOLO

model = YOLO("yolo26n.pt")

model.train(
    data="dataset1.yaml",
    epochs=50,
    imgsz=640,
    device="gpu"
)
