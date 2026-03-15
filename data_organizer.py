''' Patient Sorter
The dataset contains pictures from the same patient (sequences of about 9 images that are almost identical to each other).  
We do train/validation split randomly, so it's almost certain that nearly identical images are ending up in both sets - leakage, overfitting, yk.  
This script sorts the sequences of images into patient folders - we can then train/val split based on patients rather than individual images, 
making sure similar images aren't in both train/val sets.
'''

import os
import pandas as pd
import shutil
from tqdm import tqdm

def organize_dataset(csv_path, source_image_dir, output_dir):
    # load csv
    df = pd.read_csv(csv_path)
    check_cols = ['pathology', 'modality', 'location', 'label']
    
    # we'll be making a new patient ID folder for each sequence of images from the same patient
    patient_id = 0
    df['patient_id'] = 0
    
    for i in range(len(df)):
        if i == 0:
            df.at[i, 'patient_id'] = patient_id  # establish new patient in df
            continue
        
        # check if the csv row matches the previous row across all columns
        current_row = df.iloc[i][check_cols]
        prev_row = df.iloc[i-1][check_cols]
        
        if not current_row.equals(prev_row):  # if it doesn't match, we'll create a new patient
            patient_id += 1
            
        df.at[i, 'patient_id'] = patient_id

    # create folders and move picture
    for _, row in tqdm(df.iterrows(), total=len(df)):
        # destination data/sorted/patient_X/
        p_folder = os.path.join(output_dir, f"patient_{row['patient_id']:05d}")
        os.makedirs(p_folder, exist_ok=True)
        
        src_path = os.path.join(source_image_dir, row['filename'])
        dst_path = os.path.join(p_folder, row['filename'])
        
        if os.path.exists(src_path):
            # currently copying the image - we ought to move it so we don't duplicate, but ik Fiachra's modules are all based on data/test_images
            shutil.copy2(src_path, dst_path)
            
    # save new csv with new patient IDs
    cleaned_csv_path = csv_path.replace(".csv", "_sorted.csv")
    df.to_csv(cleaned_csv_path, index=False)
    print(f"Finished! Cleaned CSV saved to: {cleaned_csv_path}")

# run it
CSV_FILE = "data/labels.csv"
IMAGE_DIR = "data/test_images"
OUTPUT_DIR = "data/sorted"

organize_dataset(CSV_FILE, IMAGE_DIR, OUTPUT_DIR)