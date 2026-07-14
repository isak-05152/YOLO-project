from ultralytics import YOLO

# Load the pretrained model
model = YOLO(r"C:\Users\Ikbal Abdin\OneDrive\Documents\INTERNSHIP IITG\Object_Detection\runs\detect\train-8\weights\best.pt")

# Evaluate on the validation dataset
metrics = model.val(data="dataset.yaml", workers=0)

print("\n===== Evaluation Results =====")
print(f"Precision      : {metrics.box.mp:.4f}")
print(f"Recall         : {metrics.box.mr:.4f}")
print(f"mAP@0.5        : {metrics.box.map50:.4f}")
print(f"mAP@0.5:0.95   : {metrics.box.map:.4f}")
