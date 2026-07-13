from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolo26n.pt") 

model.train(
    data = "dataset.yaml",
    epochs = 50,
    imgsz = 512,
    device = 0,
    workers= 0
)
