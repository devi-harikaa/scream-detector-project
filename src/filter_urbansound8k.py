import pandas as pd
import shutil
import os
import sys

# Paths to UrbanSound8K dataset
metadata_path = "C:/Users/harik/scream-detection/src/UrbanSound8K/metadata/UrbanSound8k.csv"
input_base_dir = "C:/Users/harik/scream-detection/src/UrbanSound8K/audio/"
output_dir = "data/ambient/"

# Classes for ambient sounds
ambient_classes = ["car_horn", "engine_idling", "street_music"]

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Load metadata
try:
    metadata = pd.read_csv(metadata_path)
    print("Loaded metadata successfully.")
except FileNotFoundError:
    print(f"Error: {metadata_path} not found. Download and extract UrbanSound8K.")
    sys.exit(1)

# Filter for ambient classes and folds 1–5 (to avoid too many files)
filtered_metadata = metadata[metadata["class"].isin(ambient_classes)]
filtered_metadata = filtered_metadata[filtered_metadata["fold"].isin([1, 2, 3, 4, 5])]

# Check how many files are available
print(f"Available files: {len(filtered_metadata)}")
print(filtered_metadata["class"].value_counts())

# Copy files
count = 0
max_files = 500  # Stop at ~500 files
for _, row in filtered_metadata.iterrows():
    if count >= max_files:
        break
    # Source path (e.g., UrbanSound8K/audio/fold1/101415-1-0-0.wav)
    src = os.path.join(input_base_dir, f"fold{row['fold']}", row["slice_file_name"])
    # Destination path
    dst = os.path.join(output_dir, row["slice_file_name"])
    
    # Verify file exists
    if not os.path.exists(src):
        print(f"Warning: {src} not found, skipping.")
        continue
    
    # Copy file
    try:
        shutil.copy(src, dst)
        count += 1
        print(f"Copied {src} to {dst} ({count}/{max_files})")
    except Exception as e:
        print(f"Error copying {src}: {e}")
        continue

print(f"Copied {count} files to {output_dir}")