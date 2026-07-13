from ultralytics import YOLO

model = YOLO(
    r"runs\detect\train\weights\best.pt"
)

metrics = model.val(
    data=r"dataset.yaml"
)

print("\n===== Evaluation Results =====")
print(f"Precision      : {metrics.box.mp:.4f}")
print(f"Recall         : {metrics.box.mr:.4f}")
print(f"mAP@0.5        : {metrics.box.map50:.4f}")
print(f"mAP@0.5:0.95   : {metrics.box.map:.4f}")
