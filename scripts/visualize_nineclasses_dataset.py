import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from src.data.dataset_builder import FlexDataset3D, NineClasses3DDatasetLoader

# PROJ_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = "./OPTIMA3D_not_resampled/visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------
# Load datasets using loader
# ------------------------------
loader = NineClasses3DDatasetLoader(
    batch_size=8,
    num_workers=2,
    pin_memory=True
)

train_loader, val_loader, test_loader = loader.build_dataloaders()
train_dataset = loader.train_dataset
train_df = train_dataset.data  # pandas DataFrame

# print("Columns in CSV:", train_df.columns.tolist())
# print("Number of samples:", len(train_df))

# Validate columns
required = {"img", "label", "label_int"}
missing = required - set(train_df.columns)
if missing:
    raise ValueError(f"Missing columns: {missing}")

# ------------------------------
# Class summary
# ------------------------------
class_counts = (
    train_df
    .groupby(["label_int", "label"])
    .size()
    .reset_index(name="count")
    .sort_values("label_int")
)
# print(class_counts.to_string(index=False))

num_classes = train_df["label_int"].nunique()
# print(f"\nNumber of classes: {num_classes}")

# ------------------------------
# Visualize batch (Middle Slice)
# ------------------------------
batch = next(iter(train_loader))
imgs = batch["img"]              # [B, C, D, H, W]-> (8, 1, 49, 224, 224) /min_frames=49
labels = batch["label"]          # int
label_names = batch["label_name"]
paths = batch["path"]

fig, axs = plt.subplots(2, 4, figsize=(16, 8))
for i, ax in enumerate(axs.flatten()):
    # Extract volume: [C, D, H, W] -> [D, H, W]
    vol = imgs[i].squeeze().cpu().numpy()

    # Select the middle slice
    mid_idx = vol.shape[0] // 2 # Middle along depth dim
    img = vol[mid_idx]

    ax.imshow(img, cmap="gray")
    file_name = Path(paths[i]).name
    ax.set_title(
        f"{file_name}\nClass: {label_names[i]}",
        fontsize=9
    )
    ax.axis("off")

plt.suptitle("OPTIMA3D – Training Batch (Middle Slices)", fontsize=16)
plt.tight_layout()
save_path = os.path.join(OUTPUT_DIR, "batch_visualization_middle_slice.png")
plt.savefig(save_path, dpi=300)
print(f"\nSaved batch visualization to {save_path}")
plt.show()

# ------------------------------
# Visualize 3D Depth (One Sample Inspection)
# ------------------------------
# Plot first sample accross depth
sample_vol = imgs[0].squeeze().cpu().numpy() # [D, H, W]
depth = sample_vol.shape[0]
label = label_names[0]

# Reference: The Middle Slice
mid_idx = depth // 2
mid_slice = sample_vol[mid_idx]

# Plot every "step" number of slices
step = None
indices = np.arange(0, depth, step)

cols = 6
rows = (len(indices) + cols - 1) // cols
fig, axs = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
axs = axs.flatten()

for i, idx in enumerate(indices):
    current_slice = sample_vol[idx]
    
    # Calculate difference with reference
    # Thresholding (0.1) ignore background noise/speckle
    diff = np.clip(current_slice - mid_slice, 0, 1)
    diff[diff < 0.1] = 0 
    
    # Create RGB: Gray base with Red highlight
    r = np.clip(current_slice + diff, 0, 1)
    g = current_slice
    b = current_slice
    rgb_img = np.stack([r, g, b], axis=-1)
    
    axs[i].imshow(rgb_img)
    axs[i].set_title(f"Slice {idx}", fontsize=9)
    if idx == mid_idx:
        axs[i].set_title(f"Slice {idx} (center)", color='black', fontweight='bold')
    axs[i].axis("off")

# Hide empty subplots
for j in range(len(indices), len(axs)):
    axs[j].axis("off")

plt.suptitle(f"Volumetric Variance Inspection: {label}\n(Red highlights features not present in Middle Slice)", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

save_path_full_depth = os.path.join(OUTPUT_DIR, f"depth_visualization_{label}.png")
plt.savefig(save_path_full_depth, dpi=300)
print(f"Saved full depth inspection to {save_path_full_depth}")

# ------------------------------
# Visualize one sample per class
# ------------------------------
unique_df = (
    train_df
    .sort_values("label_int")
    .groupby("label_int", as_index=False)
    .first()
)
# print("Samples per class:")
# print(unique_df[["label_int", "label"]])

num_classes = len(unique_df)
unique_dataset = FlexDataset3D(unique_df)
unique_loader = DataLoader(
    unique_dataset,
    batch_size=num_classes,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

batch = next(iter(unique_loader))
imgs = batch["img"]
labels = batch["label"]
label_names = batch["label_name"]
paths = batch["path"]

cols = 3
rows = (num_classes + cols - 1) // cols
fig, axs = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
axs = axs.flatten()

for i in range(num_classes):
    vol = imgs[i].squeeze().cpu().numpy()
    img = vol[vol.shape[0] // 2] # Middle slice
    axs[i].imshow(img, cmap="gray")
    axs[i].set_title(f"{label_names[i]}", fontsize=10)
    axs[i].axis("off")

# Hide unused axes
for j in range(num_classes, len(axs)):
    axs[j].axis("off")

plt.suptitle("OPTIMA3D – One Sample Per Class (Middle Slices)", fontsize=16)
plt.tight_layout()
save_path = os.path.join(OUTPUT_DIR, "unique_classes.png")
plt.savefig(save_path, dpi=300)
print(f"\nSaved unique class visualization to {save_path}")
plt.show()


# ------------------------------
# Visualize one 3D sample per class MPI (Maximum Intensity Projection)
# ------------------------------

cols = 3
rows = (num_classes + cols - 1) // cols
fig, axs = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
axs = axs.flatten()

for i in range(num_classes):
    # Extract volume: [C, D, H, W] -> [D, H, W]
    vol = imgs[i].squeeze().cpu().numpy()
    
    # Calculate MIP along the Depth axis (Axis 0). 3D volume -> 2D En Face view
    mip = np.max(vol, axis=0)

    axs[i].imshow(mip, cmap="gray") # cmap="magma" or "inferno", maybe better. Standard "gray"
    
    axs[i].set_title(
        f"MIP: {label_names[i]}", 
        fontsize=10, 
        fontweight='bold'
    )
    axs[i].axis("off")

# Hide unused axes
for j in range(num_classes, len(axs)):
    axs[j].axis("off")

plt.suptitle("OPTIMA3D – En-face Maximum Intensity Projections (MIP) per Class", fontsize=16)
plt.tight_layout()

save_path_mip= os.path.join(OUTPUT_DIR, "unique_classes_mip.png")
plt.savefig(save_path_mip, dpi=300)
print(f"Saved MIP visualization to {save_path_mip}")
plt.show()


# --------------------------------------------------------------------------
# Visualize Difference: Middle Slice vs. MIP
# --------------------------------------------------------------------------

cols = 3
rows = (num_classes + cols - 1) // cols
fig, axs = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
axs = axs.flatten()

for i in range(num_classes):
    # Extract volume: [C, D, H, W] -> [D, H, W]
    vol = imgs[i].squeeze().cpu().numpy().astype(np.float32)
    
    # Get the Middle Slice
    mid_idx = vol.shape[0] // 2
    mid_slice = vol[mid_idx]
    
    # Get the MIP
    mip = np.max(vol, axis=0)
    
    # Normalize for visualization (0 to 1)
    mid_norm = (mid_slice - mid_slice.min()) / (mid_slice.max() - mid_slice.min() + 1e-8)
    mip_norm = (mip - mip.min()) / (mip.max() - mip.min() + 1e-8)
    
    # Calculate the 'Volumetric Gain' 
    gain = np.clip(mip_norm - mid_norm, 0, 1)
    # Thresholding to remove noise and highlight only significant features
    gain[gain < 0.2] = 0 
    
    # Create RGB Overlay
    # Channel 0 (Red): Middle Slice + Volumetric Gain
    # Channel 1 (Green): Middle Slice
    # Channel 2 (Blue): Middle Slice
    # Grayscale image where extra 3D info appears in Red
    r = np.clip(mid_norm + gain, 0, 1)
    g = mid_norm
    b = mid_norm
    
    rgb_overlay = np.stack([r, g, b], axis=-1)
    
    axs[i].imshow(rgb_overlay)
    axs[i].set_title(
        f"{label_names[i]}", 
        fontsize=11, fontweight='bold'
    )
    axs[i].axis("off")

# Hide unused axes
for j in range(num_classes, len(axs)):
    axs[j].axis("off")

plt.suptitle("3D Advantage: Volumetric Details (Red) vs. Middle Slice (Gray)", fontsize=18, y=0.98)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

save_path_diff = os.path.join(OUTPUT_DIR, "3d_context_difference.png")
plt.savefig(save_path_diff, dpi=300, bbox_inches='tight')
print(f"Saved Difference Map visualization to {save_path_diff}")
plt.show()


# --------------------------------------------------------------------------
# Layered Volume
# --------------------------------------------------------------------------
def plot_layered_volume(volume, label_name, num_slices=10, spacing=2.0, save_name="layered_horiz.png"):
    """
    Plots slices stacked horizontally (Left-to-Right).
    spacing: multiplier to increase distance between slices for clarity.
    """
    depth, height, width = volume.shape
    indices = np.linspace(0, depth - 1, num_slices).astype(int)
    
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create grids for Height and Width
    y_grid, z_grid = np.meshgrid(np.arange(height), np.arange(width))

    for i, idx in enumerate(indices):
        slice_data = volume[idx, :, :].astype(float).T # Transpose to match grid
        slice_norm = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min() + 1e-8)
        
        # Position each slice at a 'Horizontal' X position
        x_pos = i * (width * spacing)
        
        ax.plot_surface(np.full_like(y_grid, x_pos), y_grid, z_grid,
                        rstride=2, cstride=2, 
                        facecolors=plt.cm.gray(slice_norm), 
                        shade=False, 
                        alpha=0.6, # Transparency value
                        antialiased=True)

    ax.set_title(f"Horizontal 3D Stack: {label_name}", fontsize=15)
    ax.set_xlabel("Depth progression (Slices)")
    ax.set_ylabel("Height (Y)")
    ax.set_zlabel("Width (X)")
    
    # Perspective to see them side-by-side
    ax.view_init(elev=15, azim=-60)
    
    # remove background grids
    ax.grid(False)
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved layered volume visualization to {save_path}")


# Pick the first sample from last batch
sample_vol = imgs[0].squeeze().cpu().numpy()
sample_label = label_names[0]
plot_layered_volume(sample_vol,
                    sample_label, 
                    num_slices=12, 
                    spacing=2.0,
                    save_name=f"layered_stack_{sample_label}.png")

