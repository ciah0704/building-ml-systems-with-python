import os

DATA_DIR = "data" # put your Posts.xml into this directory
CHART_DIR = "charts"

filtered = os.path.join(DATA_DIR, "filtered.tsv")
filtered_meta = os.path.join(DATA_DIR, "filtered-meta.json")

chosen = os.path.join(DATA_DIR, "chosen.tsv")
chosen_meta = os.path.join(DATA_DIR, "chosen-meta.json")