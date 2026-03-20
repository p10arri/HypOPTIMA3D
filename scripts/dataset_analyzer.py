import os
import sys
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import ast

# --- CONFIG ---
DATA_DIR = "OPTIMA3D_not_resampled/processed/nineclasses3D"
SPLITS = {"train": "train.csv", "val": "val.csv", "test": "test.csv"}
OUT_DIR = "./OPTIMA3D_not_resampled/analysis"
os.makedirs(OUT_DIR, exist_ok=True)

# --- LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(OUT_DIR, "analysis.log")), 
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def parse_shape_string(val):
    """
    Parses a string like '(49, 496, 512)' into a tuple of ints (49, 496, 512).
    Returns (0, 0, 0) if parsing fails.
    """
    if pd.isna(val) or str(val).strip() == "":
        return 0, 0, 0
    try:
        # Try literal eval first
        shape = ast.literal_eval(str(val))
        if isinstance(shape, (list, tuple)) and len(shape) == 3:
            return tuple(map(int, shape))
    except:
        # Fallback to regex if string is malformed
        nums = re.findall(r'\d+', str(val))
        if len(nums) >= 3:
            return int(nums[0]), int(nums[1]), int(nums[2])
    return 0, 0, 0

def load_split(name, path):
    logger.info(f"Loading {name}: {path}")
    df = pd.read_csv(path)
    
    # 1. Parse Original Shapes
    if "shape_original" in df.columns:
        shapes = df["shape_original"].apply(parse_shape_string)
        df["orig_depth"] = shapes.apply(lambda x: x[0])
        df["orig_height"] = shapes.apply(lambda x: x[1])
        df["orig_width"] = shapes.apply(lambda x: x[2])
        # Calculate total voxels (size metric)
        df["total_voxels"] = df["orig_depth"] * df["orig_height"] * df["orig_width"]
    
    # 2. Parse Processed Depth
    if "shape" in df.columns:
        df["n_frames"] = df["shape"].apply(lambda x: parse_shape_string(x)[0])
    
    # 3. Handle Vendor mapping
    if "manufacturer" in df.columns:
        df["vendor"] = df["manufacturer"].fillna("Unknown")
    else:
        df["vendor"] = "Unknown"
        
    return df

def analyze_shapes(df, name):
    """Compares original volume dimensions across labels and vendors."""
    if "orig_depth" not in df.columns:
        return

    logger.info(f"--- {name.upper()} : SHAPE & SIZE ANALYSIS ---")
    
    # Define metrics to plot
    metrics = ['orig_depth', 'orig_height', 'orig_width']
    
    # 1. Compare Dimensions by Label
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for i, metric in enumerate(metrics):
        sns.boxplot(data=df, x='label', y=metric, ax=axes[i], palette="viridis", hue='label', legend=False)
        axes[i].set_title(f"{metric.replace('orig_', '').capitalize()} by Label")
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.suptitle(f"{name.capitalize()} Split: Original Dimensions by Disease Label", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{name}_shape_by_label.png"))
    plt.close()

    # 2. Compare Dimensions by Vendor
    if df["vendor"].nunique() > 1:
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        for i, metric in enumerate(metrics):
            sns.boxplot(data=df, x='vendor', y=metric, ax=axes[i], palette="magma", hue='vendor', legend=False)
            axes[i].set_title(f"{metric.replace('orig_', '').capitalize()} by Vendor")
            axes[i].tick_params(axis='x', rotation=45)
        
        plt.suptitle(f"{name.capitalize()} Split: Original Dimensions by Vendor", fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"{name}_shape_by_vendor.png"))
        plt.close()

def analyze_vendors(df, name):
    """Class Distribution Heatmap per Vendor."""
    if df["vendor"].nunique() <= 1 and "Unknown" in df["vendor"].unique():
        return

    logger.info(f"--- {name.upper()} : VENDOR BIAS ANALYSIS ---")
    ct = pd.crosstab(df["vendor"], df["label"])
    
    if not ct.empty:
        plt.figure(figsize=(12, 7))
        sns.heatmap(ct, annot=True, fmt="d", cmap="YlGnBu")
        plt.title(f"{name.capitalize()} - Label Distribution by Vendor")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"{name}_vendor_bias.png"))
        plt.close()

def main():
    for name, file in SPLITS.items():
        path = os.path.join(DATA_DIR, file)
        if not os.path.exists(path):
            continue
    
        df = load_split(name, path)
        
        # LOG BASIC INFO
        logger.info(f"--- {name.upper()} : BASIC STATS ---")
        logger.info(f"Samples: {len(df)} | Vendors: {df['vendor'].unique().tolist()}")
        
        # RUN ANALYSES
        analyze_shapes(df, name)
        analyze_vendors(df, name)

    logger.info(f"Analysis complete. Results saved in: {OUT_DIR}")

if __name__ == "__main__":
    main()