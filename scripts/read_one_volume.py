import numpy as np

file_path = "OPTIMA3D/processed/nineclasses3D/TEST_RUN/train/volume_145.npz"

data = np.load(file_path, allow_pickle=True)

print("\n=== FULL NPZ INSPECTION ===\n")

for key in data.files:
    print(f"\n===== {key} =====")
    arr = data[key]

    print(f"type: {type(arr)}")
    print(f"dtype: {arr.dtype}")
    print(f"shape: {arr.shape}")

    #  min/max on numeric types
    if np.issubdtype(arr.dtype, np.number) and arr.size > 0:
        print(f"min: {arr.min()}")
        print(f"max: {arr.max()}")
    else:
        # Print strings
        print(f"value: {arr}")

    # Handle metadata stored as 0-d arrays
    if arr.ndim == 0:
        print(f"unpacked value: {arr.item()}")

    # Show sample slice for volumes (D, H, W)
    if arr.ndim >= 3:
        print(f"sample slice [0, :20, :20] (top-left corner):")
        # Printing just a small corner
        print(arr[0, :20, :20])

print("\n=== INSPECTION COMPLETE ===")