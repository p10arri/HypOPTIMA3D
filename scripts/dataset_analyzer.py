import os
import sys
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# CONFIG
DATA_DIR = "OPTIMA3D/processed/nineclasses3D"

SPLITS = {
    "train": "train.csv",
    "val": "val.csv",
    "test": "test.csv"
}

FULL_DATASET_PATH = ".private/OPTIMA_15abr_all_vendors.csv"


OUT_DIR = "./OPTIMA3D/analysis"
os.makedirs(OUT_DIR, exist_ok=True)


LOG_PATH = os.path.join(OUT_DIR, "dataset_analysis.log")


# LOGGING
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger(__name__)



def load_split(name, path):
    logger.info(f"Loading {name}: {path}")
    df = pd.read_csv(path)
    logger.info(f"{name} samples: {len(df)}")
    return df

def load_full_dataset(path):
    logger.info(f"Loading FULL dataset: {path}")
    df = pd.read_csv(path)
    # Extract vendor from sample_path
    if "sample_path" in df.columns:
        df["vendor"] = df["sample_path"].apply(lambda x: x.split("/")[3])
    else:
        logger.warning("Column 'sample_path' not found in full dataset. Vendor analysis will be skipped.")

    logger.info(f"Full dataset samples: {len(df)}")
    return df

def analyze_basic(df, name):
    """Basic dataset statistics"""

    logger.info(f"--- {name.upper()} : BASIC STATS ---")

    n_samples = len(df)
    n_patients = df["FileSetId"].nunique()
    n_classes = df["label"].nunique()

    logger.info(f"Samples   : {n_samples}")
    logger.info(f"Patients  : {n_patients}")
    logger.info(f"Classes   : {n_classes}")

    return {
        "samples": n_samples,
        "patients": n_patients,
        "classes": n_classes
    }


def analyze_classes(df, name):
    """Class distribution"""

    logger.info(f"--- {name.upper()} : CLASS DISTRIBUTION ---")

    counts = df["label"].value_counts().sort_index()
    percent = 100 * counts / len(df)

    table = pd.DataFrame({
        "samples": counts,
        "percentage": percent.round(2)
    })

    logger.info(f"\n{table}")

    # Save
    table.to_csv(
        os.path.join(OUT_DIR, f"{name}_class_distribution.csv")
    )

    # Plot
    plt.figure()
    counts.plot(kind="bar")
    plt.title(f"{name.capitalize()} - Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Samples")
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(
        os.path.join(OUT_DIR, f"{name}_class_distribution.png"),
        dpi=300
    )
    plt.close()

    return table


def analyze_patients(df, name):
    """Patients per class"""

    logger.info(f"--- {name.upper()} : PATIENTS PER CLASS ---")

    patients = (
        df.groupby("label")["FileSetId"]
        .nunique()
        .sort_index()
    )

    logger.info(f"\n{patients}")

    patients.to_csv(
        os.path.join(OUT_DIR, f"{name}_patients_per_class.csv")
    )

    # Plot
    plt.figure()
    patients.plot(kind="bar")
    plt.title(f"{name.capitalize()} - Patients per Class")
    plt.xlabel("Class")
    plt.ylabel("Patients")
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(
        os.path.join(OUT_DIR, f"{name}_patients_per_class.png"),
        dpi=300
    )
    plt.close()

    return patients


def analyze_slices(df, name):
    """n_frames statistics"""

    logger.info(f"--- {name.upper()} : SLICE STATS ---")

    stats = {
        "min": df["n_frames"].min(),
        "max": df["n_frames"].max(),
        "mean": df["n_frames"].mean(),
        "median": df["n_frames"].median(),
        "std": df["n_frames"].std()
    }

    for k, v in stats.items():
        logger.info(f"{k:>8}: {v:.2f}")

    # Histogram
    plt.figure()
    plt.hist(df["n_frames"], bins=30)
    plt.title(f"{name.capitalize()} - Slice Distribution")
    plt.xlabel("Slices")
    plt.ylabel("Frequency")
    plt.tight_layout()

    plt.savefig(
        os.path.join(OUT_DIR, f"{name}_n_frames_hist.png"),
        dpi=300
    )
    plt.close()

    # Per-class stats
    per_class = (
        df.groupby("label")["n_frames"]
        .agg(["min", "max", "mean", "median", "std"])
        .round(2)
    )

    logger.info("\nSlice stats per class:")
    logger.info(f"\n{per_class}")

    per_class.to_csv(
        os.path.join(OUT_DIR, f"{name}_slice_stats_per_class.csv")
    )

    # Boxplot
    plt.figure(figsize=(10, 5))
    df.boxplot(column="n_frames", by="label", grid=False)
    plt.title(f"{name.capitalize()} - Slices by Class")
    plt.suptitle("")
    plt.xlabel("Class")
    plt.ylabel("Slices")
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(
        os.path.join(OUT_DIR, f"{name}_n_frames_boxplot.png"),
        dpi=300
    )
    plt.close()

    return stats, per_class




def analyze_split(name, path):

    logger.info("=" * 70)
    logger.info(f"ANALYZING {name.upper()}")
    logger.info("=" * 70)

    df = load_split(name, path)
    analyze_basic(df, name)
    analyze_classes(df, name)
    analyze_patients(df, name)
    analyze_slices(df, name)

    logger.info(f"Finished {name}\n")

def check_data_leakage(split_dict):
    """Ensure no patient (FileSetId) is shared across Train, Val, and Test"""
    logger.info("=" * 70)
    logger.info("CHECKING FOR DATA LEAKAGE")
    logger.info("=" * 70)
    
    train_df = split_dict.get("train")
    val_df = split_dict.get("val")
    test_df = split_dict.get("test")

    if train_df is None or val_df is None or test_df is None:
        logger.warning("Skipping leakage check: Not all splits available.")
        return

    sets = {
        "Train-Val": (set(train_df["FileSetId"]), set(val_df["FileSetId"])),
        "Train-Test": (set(train_df["FileSetId"]), set(test_df["FileSetId"])),
        "Val-Test": (set(val_df["FileSetId"]), set(test_df["FileSetId"]))
    }

    clean = True
    for pair_name, (set_a, set_b) in sets.items():
        overlap = set_a.intersection(set_b)
        if overlap:
            logger.error(f"LEAKAGE FOUND in {pair_name}: {len(overlap)} shared patients!")
            logger.error(f"Shared IDs (first 5): {list(overlap)[:5]}")
            clean = False
    
    if clean:
        logger.info("CLEAN: No patient overlap found between splits. Data is valid.")

def analyze_vendors(df, name):
    """Check if certain classes are tied to specific hardware vendors/sites"""
    if "vendor" not in df.columns:
        logger.error(f"Cannot run Vendor Analysis for {name}: 'vendor' column missing.")
        return
    logger.info(f"--- {name.upper()} : VENDOR/BIAS ANALYSIS ---")
    
    vendor_stats = pd.crosstab(df["vendor"], df["label"])
    logger.info(f"\n{vendor_stats}")
    
    # Heatmap for the thesis report
    plt.figure(figsize=(12, 7))
    sns.heatmap(vendor_stats, annot=True, fmt="d", cmap="YlGnBu")
    plt.title(f"{name.capitalize()} - Class Distribution by Vendor Site")
    plt.ylabel("Source Folder / Vendor")
    plt.xlabel("Pathology Class")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{name}_vendor_bias.png"), dpi=300)
    plt.close()

def analyze_scan_dimensions(df, name, target_depth=49):
    """Verify that scans have enough slices for the 3D target window"""
    logger.info(f"--- {name.upper()} : 3D SCAN CONSISTENCY ---")
    
    too_small = df[df["n_frames"] < target_depth]
    
    if not too_small.empty:
        logger.warning(f"Found {len(too_small)} scans with < {target_depth} frames.")
        logger.info(f"Small scans distribution:\n{too_small['label'].value_counts()}")
    else:
        logger.info(f"SUCCESS: All scans meet the target depth of {target_depth}.")

def main():
    logger.info("Starting dataset analysis")

    if not os.path.exists(FULL_DATASET_PATH):
        logger.error(f"CRITICAL: Full dataset not found at {FULL_DATASET_PATH}. Vendor analysis will fail.")
        return

    df_full = load_full_dataset(FULL_DATASET_PATH)
    
    # Map: FileSetId -> Vendor
    vendor_map = df_full.drop_duplicates("FileSetId").set_index("FileSetId")["vendor"].to_dict()

    loaded_dataframe = {}

    # Analyze splits
    for name, file in SPLITS.items():
        path = os.path.join(DATA_DIR, file)
        if not os.path.exists(path):
            logger.error(f"Missing file: {path}")
            continue
    
        df = load_split(name, path)

        # MAP VENDOR FROM FULL DATASET
        if "FileSetId" in df.columns:
            df["vendor"] = df["FileSetId"].map(vendor_map)
            # Fill missing if some IDs weren't in the full CSV
            df["vendor"] = df["vendor"].fillna("unknown")
        else:
            logger.warning(f"FileSetId not found in {name}.csv, cannot map vendors.")

        loaded_dataframe[name] = df
        
        logger.info("=" * 70)
        logger.info(f"ANALYZING {name.upper()}")
        logger.info("=" * 70)

        analyze_basic(df, name)
        analyze_classes(df, name)
        analyze_patients(df, name)
        analyze_slices(df, name)
        analyze_vendors(df, name)       
        analyze_scan_dimensions(df, name)

    # Check leakage between splits
    check_data_leakage(loaded_dataframe)
    
    logger.info("=" * 70)
    logger.info("FINAL FULL DATASET SUMMARY")
    analyze_basic(df_full, "full")
    analyze_vendors(df_full, "full")

    logger.info("All analyses completed")
    logger.info(f"Outputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()