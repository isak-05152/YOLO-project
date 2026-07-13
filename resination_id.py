import os

# Original VisDrone class -> New class
class_map = {
    0: 0,  # pedestrian
    1: 1,  # people/person
    3: 2   # car
}

folders = [
    r"path to training_dataset",
    r"path to validation_dataset"
]

for folder in folders:

    for filename in os.listdir(folder):

        if not filename.endswith(".txt"):
            continue

        file_path = os.path.join(folder, filename)

        new_lines = []

        with open(file_path, "r") as file:

            for line in file:

                parts = line.strip().split()

                if len(parts) < 5:
                    continue

                old_class = int(parts[0])

                if old_class in class_map:

                    parts[0] = str(class_map[old_class])

                    new_lines.append(" ".join(parts))

        with open(file_path, "w") as file:
            file.write("\n".join(new_lines))

print("Dataset filtered successfully!")
print("0 = pedestrian")
print("1 = people")
print("2 = car")
