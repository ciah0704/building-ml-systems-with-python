import os

# DATA_DIR = r"C:\pymlbook-data\ch05"
DATA_DIR = "Building_ML_Systems_with_Python/chapter_05_Codes/data"
CHART_DIR = os.path.join("..", "charts")

filtered = os.path.join(DATA_DIR, "filtered.tsv")
filtered_meta = os.path.join(DATA_DIR, "filtered-meta.json")

chosen = os.path.join(DATA_DIR, "chosen.tsv")
chosen_meta = os.path.join(DATA_DIR, "chosen-meta.json")
