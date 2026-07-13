from ultralytics import YOLO

model = YOLO("yolo26l.pt") 

results = model(source="/home/isak/yolo_env/training", classes=[0,2,9], conf=0.25, show=True, save=True)

for result in results:
    for box in result.boxes:
        class_id = int(box.cls[0])
        
        if class_id not in [0,2,9]:
            continue
        confidence = float(box.conf[0])    

        print(
            f"Detected: {model.names[class_id]}, "
            f"Confidence: {confidence:.2f}"
        )
