import os

data_path = "data/raw/"

for folder in os.listdir(data_path):
    full_path = os.path.join(data_path, folder)
    print(folder, "→ contains:", os.listdir(full_path))
