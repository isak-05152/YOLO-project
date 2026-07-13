import os
import json

# ======================================================
# CHANGE THIS TO YOUR UAVDT DATASET ROOT
# ======================================================
DATASET_ROOT = r"images/UAVDT_sample"
# YOLO class mapping
CLASS_MAP = {
    "car": 0,
    "vehicle": 1,
    "truck": 2,
    "bus": 3
}


def convert_json(json_path, output_path):
    """
    Convert one JSON annotation file into YOLO format.
    """

    with open(json_path, "r") as f:
        data = json.load(f)

    img_width = data["size"]["width"]
    img_height = data["size"]["height"]

    yolo_lines = []

    for obj in data["objects"]:

        class_name = obj["classTitle"].lower()

        if class_name not in CLASS_MAP:
            continue

        class_id = CLASS_MAP[class_name]

        exterior = obj["points"]["exterior"]

        x1, y1 = exterior[0]
        x2, y2 = exterior[1]

        xmin = min(x1, x2)
        xmax = max(x1, x2)

        ymin = min(y1, y2)
        ymax = max(y1, y2)

        width = xmax - xmin
        height = ymax - ymin

        center_x = xmin + width / 2
        center_y = ymin + height / 2

        center_x /= img_width
        center_y /= img_height
        width /= img_width
        height /= img_height

        yolo_lines.append(
            f"{class_id} "
            f"{center_x:.6f} "
            f"{center_y:.6f} "
            f"{width:.6f} "
            f"{height:.6f}"
        )

    with open(output_path, "w") as f:
        f.write("\n".join(yolo_lines))


def convert_dataset(split):

    image_folder = os.path.join(DATASET_ROOT, split, "img")
    annotation_folder = os.path.join(DATASET_ROOT, split, "ann")
    label_folder = os.path.join(
        "labels",
        "UAVDT_sample",
        split,
        "img")

    os.makedirs(label_folder, exist_ok=True)

    image_extensions = [".jpg", ".jpeg", ".png"]

    for image_name in os.listdir(image_folder):

        if not image_name.lower().endswith(tuple(image_extensions)):
            continue

        json_name = image_name + ".json"

        json_path = os.path.join(annotation_folder, json_name)

        if not os.path.exists(json_path):
            print(f"Missing annotation : {json_name}")
            continue

        output_path = os.path.join(
            label_folder,
            os.path.splitext(image_name)[0] + ".txt"
        )

        convert_json(json_path, output_path)

    print(f"{split} conversion finished.")


def main():

    splits = [
        "train",
        "test"
    ]

    for split in splits:
        convert_dataset(split)

    print("\n==============================")
    print("Conversion Completed")
    print("==============================")


if __name__ == "__main__":
    main()
