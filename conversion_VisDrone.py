import os
from PIL import Image

# ==========================
# CHANGE THESE PATHS
# ==========================

IMAGE_DIR = "/home/isak/yolo_env/VisDrone2019-DET-train/images"
ANNOTATION_DIR = "/home/isak/yolo_env/VisDrone2019-DET-train/annotations"
LABEL_DIR = "/home/isak/yolo_env/VisDrone2019-DET-train/labels"

os.makedirs(LABEL_DIR, exist_ok=True)

# VisDrone class mapping
# Ignore regions (0) and "others" (11) are skipped
CLASS_MAP = {
    1: 0,   # pedestrian
    2: 1,   # people
    3: 2,   # bicycle
    4: 3,   # car
    5: 4,   # van
    6: 5,   # truck
    7: 6,   # tricycle
    8: 7,   # awning-tricycle
    9: 8,   # bus
    10: 9   # motor
}

for ann_file in os.listdir(ANNOTATION_DIR):
    if not ann_file.endswith(".txt"):
        continue

    image_name = ann_file.replace(".txt", ".jpg")
    image_path = os.path.join(IMAGE_DIR, image_name)

    if not os.path.exists(image_path):
        image_name = ann_file.replace(".txt", ".png")
        image_path = os.path.join(IMAGE_DIR, image_name)

    if not os.path.exists(image_path):
        print(f"Image not found: {ann_file}")
        continue

    img = Image.open(image_path)
    img_width, img_height = img.size

    output_lines = []

    with open(os.path.join(ANNOTATION_DIR, ann_file), "r") as f:
        for line in f:
            parts = line.strip().split(",")

            if len(parts) < 8:
                continue

            x = float(parts[0])
            y = float(parts[1])
            w = float(parts[2])
            h = float(parts[3])

            score = int(parts[4])
            category = int(parts[5])

            # Ignore ignored regions and "others"
            if category not in CLASS_MAP:
                continue

            class_id = CLASS_MAP[category]

            # Convert to YOLO format
            x_center = (x + w / 2) / img_width
            y_center = (y + h / 2) / img_height
            width = w / img_width
            height = h / img_height

            output_lines.append(
                f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            )

    label_path = os.path.join(LABEL_DIR, ann_file)

    with open(label_path, "w") as f:
        f.write("\n".join(output_lines))

print("Conversion completed successfully!")
