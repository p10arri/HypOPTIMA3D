import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pydicom as dicom
from tqdm import tqdm

TRAIN_CSV = "OPTIMA3D_not_resampled/processed/nineclasses3D/train.csv"
ALL_VENDORS_CSV = ".private/OPTIMA_15abr_all_vendors.csv"
OUTPUT_DIR = "OPTIMA3D_not_resampled/visualizations/volume_comparison"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def plot_slice_comparison(vol_orig, vol_proc, label_name, save_path):
    """
    Plots the middle 5 slices of both volumes for a direct quality comparison.
    """
    # Find middle indices
    mid_orig = vol_orig.shape[0] // 2
    mid_proc = vol_proc.shape[0] // 2
    
    # Select 5 slices centered at the middle
    indices_orig = np.arange(mid_orig - 2, mid_orig + 3)
    indices_proc = np.arange(mid_proc - 2, mid_proc + 3)
    
    fig, axes = plt.subplots(2, 5, figsize=(25, 10), facecolor='black')
    plt.subplots_adjust(wspace=0.05, hspace=0.2)

    for i in range(5):
        #  Original Slices (Top Row) 
        idx_o = indices_orig[i]
        if 0 <= idx_o < vol_orig.shape[0]:
            slice_o = vol_orig[idx_o, :, :]
            axes[0, i].imshow(slice_o, cmap='gray', aspect='auto')
            axes[0, i].set_title(f"Orig Slice {idx_o}", color='white', fontsize=10)
        axes[0, i].axis('off')

        #  Processed Slices (Bottom Row)
        idx_p = indices_proc[i]
        if 0 <= idx_p < vol_proc.shape[0]:
            slice_p = vol_proc[idx_p, :, :]
            axes[1, i].imshow(slice_p, cmap='gray', aspect='auto')
            axes[1, i].set_title(f"Resampled Slice {idx_p}", color='white', fontsize=10)
        axes[1, i].axis('off')

    plt.suptitle(f"Central Volume Comparison: {label_name.upper()}\nTop: Original | Bottom: Resampled", 
                 color='white', fontsize=20, y=0.98)
    
    plt.savefig(save_path, dpi=200, facecolor='black', bbox_inches='tight')
    plt.close(fig)

def main():
    df_train = pd.read_csv(TRAIN_CSV)
    df_vendors = pd.read_csv(ALL_VENDORS_CSV)
    
    # Get one sample per class
    unique_classes = df_train.groupby("label").first().reset_index()

    for _, row in tqdm(unique_classes.iterrows(), total=len(unique_classes)):
        label = row['label']
        fileset_id = row['FileSetId']
        processed_path = row['img']

        try:
            with np.load(processed_path) as data:
                vol_proc = data['img'] 
        except Exception as e:
            print(f"Error NPZ {label}: {e}")
            continue

        # Find and Load Original DICOM
        vendor_row = df_vendors[df_vendors['FileSetId'] == fileset_id]
        if vendor_row.empty: continue
        
        dicom_path = vendor_row.iloc[0]['sample_path']
        try:
            dcm = dicom.dcmread(dicom_path, force=True)
            dcm.SamplesPerPixel = 1 # The critical fix
            vol_orig = dcm.pixel_array
        except Exception as e:
            print(f"Error DICOM {label}: {e}")
            continue

        # 3. Render and Save
        save_name = f"volume_comparison_{label}.png"
        save_path = os.path.join(OUTPUT_DIR, save_name)
        plot_slice_comparison(vol_orig, vol_proc, label, save_path)
        print(f"Saved: {save_path}")

if __name__ == "__main__":
    main()