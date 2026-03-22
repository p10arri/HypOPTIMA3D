import os
import csv
import torch
import cv2
import numpy as np
import ast
import pandas as pd
import pydicom as dicom
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, List, Union, Optional, Callable, Any
from abc import ABC, abstractmethod

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode
from sklearn.model_selection import StratifiedKFold
from scipy.ndimage import zoom

from src.utils.enums import DatasetSplit, NineClassesLabel

PROJ_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path("/optima/data")

DATA_DIR_OPTIMA3D = PROJ_ROOT / "OPTIMA3D"

PROCESSED_DATA_DIR = DATA_DIR_OPTIMA3D / "processed"
PRIVATE_FOLDER =  PROJ_ROOT / ".private"
CSV_FILE = PRIVATE_FOLDER / "OPTIMA_15abr_all_vendors.csv"

OUTPUT_COLUMNS = ["img", "FileSetId", "label", "label_int", "shape", "shape_original", "manufacturer"]

def merge_full_dataset(output_path=CSV_FILE):
    if os.path.exists(output_path):
        print(f"Dataset already exists: {output_path}")
        return

    directory = output_path.parent
    basename = output_path.stem
    files = list(directory.glob(f"*{basename}*"))
    
    combined_df = pd.concat(map(pd.read_csv, files), ignore_index=True)
    combined_df.to_csv(output_path, index=False)
    print(f"Full merge dataset saved to: {output_path}")

def resample_volume(volume, spacing, target_spacing=(0.05, 0.01, 0.01)):
    """
    Volume Inter-Slice and Intra-Slice spacing is not standarized between manufacturers. This function resamples the volumes to a fixed volume spacing.
    Trilinear interpolation is used. Thus, after resampling each voxel represents the same physical size across volumes.

    """
    z, y, x = spacing

    if z is None or y is None or x is None:
        raise ValueError("Missing spacing")

    scale_factors = (
        z / target_spacing[0],
        y / target_spacing[1],
        x / target_spacing[2]
    )

    resampled = zoom(volume, zoom=scale_factors, order=1)  # trilinear interpolation
    return resampled

class DatasetSaver(ABC):
    def __init__(self):
        self.dataset_name = self._get_dataset_name()
        self.transforms_npz = self._get_transform()

    def check_preprocessed_data_saved(self) -> bool:
        base = PROCESSED_DATA_DIR / self.dataset_name
        # Use DatasetSplit enum for check
        return all((base / f"{split.value}.csv").exists() for split in DatasetSplit)

    def _make_new_csv_file(self, name: Path):
        with open(name.with_suffix(".csv"), mode='w') as file:
            writer = csv.writer(file)
            writer.writerow(OUTPUT_COLUMNS)

    def _append_to_csv_file(self, name: Path, row: pd.Series):
        with open(name.with_suffix(".csv"), mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([row.get(col, "") for col in OUTPUT_COLUMNS])

    def save_dataset(self):
        *data_splits, min_frames = self._load_raw_data() # (train_df, val_df, test_df)
        
        for split_enum, df in zip(DatasetSplit, data_splits):
            save_path = self._get_save_path() / split_enum.value
            os.makedirs(save_path, exist_ok=True)

            csv_name = self._get_save_path() / split_enum.value
            self._make_new_csv_file(csv_name)
            
            with tqdm(df.iterrows(), total=len(df), desc=f"Processing {split_enum.name} set") as pbar:
                for i, row in pbar:
                    image_path = row["sample_path"]

                    try:
                        # Get 3d crop of min_frames
                        img, metadata = self._get_image_3d(image_path, min_frames)

                        img_name = f"{save_path}/volume_{i+1}.npz"

                        row_out = row.copy()
                        row_out["img"] = str(img_name)
                        row_out["shape"] = metadata.get("shape")
                        row_out["shape_original"] = metadata.get("shape_original")
                        row_out["manufacturer"] = metadata.get("manufacturer")
                        row_out["n_frames"] = min_frames
                        
                        self._append_to_csv_file(csv_name, row_out)
                        # Compressed 3D volumes
                        np.savez_compressed(
                            img_name,
                            img=img,
                            **{k: np.array(v) if not isinstance(v, (str, int, float, np.ndarray)) else v
                            for k, v in metadata.items()}
                        )
                    except Exception as e:
                        pbar.write(f"Warning: Skipping corrupted file {image_path} | Error: {e}")
                        continue

    @abstractmethod
    def _get_transform(self): pass
    @abstractmethod
    def _get_dataset_name(self) -> str: pass
    @abstractmethod
    def _load_raw_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: pass
    @abstractmethod
    def _get_image_3d(self, path: str, target_frames: int): pass
    @abstractmethod
    def _get_save_path(self) -> Path: pass

class NineClasses3DDatasetSaver(DatasetSaver):
    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)

        super().__init__()

    def _get_dataset_name(self) -> str:
        return "nineclasses3D"

    def _load_raw_data(self):
        random_state = 42
        df = pd.read_csv(self.csv_path)

        missing_mask = ~df["sample_path"].apply(os.path.exists)
        df_valid = df.loc[~missing_mask].reset_index(drop=True)

        
        # Get the minimum number of frames
        min_frames = int(df_valid["n_frames"].min())

        sgkf = StratifiedKFold(n_splits=4, shuffle=True, random_state=random_state)
        train_idx, val_idx = next(sgkf.split(df_valid, y=df_valid["label_int"]))
        
        train_df = df_valid.iloc[train_idx].reset_index(drop=True)
        val_df = df_valid.iloc[val_idx].reset_index(drop=True)

        test_size = int(0.1 * len(train_df))
        test_df = train_df.sample(test_size, random_state=random_state)
        train_df = train_df.drop(test_df.index).reset_index(drop=True)
        
        return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True), min_frames

    def _get_image_3d(self, path: str, target_frames: int):
        dcm  = dicom.dcmread(path, force=True)

        #  FIX MISSING TAGS FOR NON-STANDARD DICOMS 
        # Manually add the missing 'Samples per Pixel' tag
        if 'SamplesPerPixel' not in dcm:
            dcm .SamplesPerPixel = 1
        # Some files might also miss 'PlanarConfiguration' or 'PhotometricInterpretation'
        if 'PhotometricInterpretation' not in dcm :
            dcm .PhotometricInterpretation = "MONOCHROME2"

        # Extract spacing
        z = getattr(dcm, "SpacingBetweenSlices", None) or getattr(dcm, "SliceThickness", None)
        yx = getattr(dcm, "PixelSpacing", [None, None])

        z = float(z) if z is not None else None
        y, x = map(float, yx) if yx is not None and len(yx) == 2 else (None, None)

        spacing = (z, y, x)

        # Load volume
        volume = dcm.pixel_array # (Slices, H, W)
        
        # Resampling
        volume_resampled = resample_volume(volume, spacing=spacing, target_spacing=(0.05, 0.01, 0.01))

        # Crop center in depth (Z axis)
        total_frames = volume_resampled.shape[0] # the center is now in physical space, not original space
        center = total_frames // 2
        half_crop = target_frames // 2

        start = max(0, center-half_crop)
        end = start + target_frames

        # Adjust for edge cases
        if end >  total_frames:
            end = total_frames
            start = max(0, end - target_frames)
        
        volume_crop = volume_resampled[start:end, : , :]

        # Pad if needed
        if volume_crop.shape[0] < target_frames:
            pad_val = target_frames - volume_crop.shape[0]
            volume_crop = np.pad(volume_crop, ( (0, pad_val), (0,0), (0,0) ),mode="edge" )

        # Normalize 
        # NOTE: Per volume normalization (min-max) may introduce variability across scans, better percentile normalization
        p1, p99 = np.percentile(volume_crop, (1, 99))
        volume_crop = np.clip(volume_crop, p1, p99)
        volume_crop = ((volume_crop - p1) / (p99 - p1 + 1e-8) * 255).astype(np.uint8)


        metadata =  {
            "spacing": (0.05, 0.01, 0.01),
            "spacing_original": spacing, 
            "shape": volume_crop.shape,
            "shape_resampled": volume_resampled.shape,
            "shape_original": volume.shape,
            "n_frames_used": target_frames,
            "manufacturer": getattr(dcm, "Manufacturer", "UNKNOWN")
        }
        return volume_crop, metadata

    def _get_save_path(self):
        return Path(PROCESSED_DATA_DIR) / self.dataset_name
    
    def _get_transform(self):
        return lambda x: x    

class FlexDataset3D(Dataset):
    def __init__(self, data: pd.DataFrame, transform: Optional[Union[Callable, transforms.Compose]] = None):
        self.data = data
        self.transform = transform if transform else self.get_transform()

    def __len__(self):
        return len(self.data)
    
    def get_transform(self):
        # Normalize the tensor
        return lambda x: x.float() / 255.0

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        path = row["img"]
        
        # Get class
        label_int = int(row["label_int"])
        label_enum = NineClassesLabel(label_int)

        # Load 3d image
        sample = np.load(path)["img"] # (D,H,W)
        # Convert to float tensor and add grayscale channel dim:(1,D,H,W)
        sample = torch.from_numpy(sample).unsqueeze(0)

        sample = self.transform(sample)

        def process_tensor(img_tensor):
            # Normalize
            if img_tensor.max() > 1.0:
                img_tensor = img_tensor.float() / 255.0
                
            # Spatial resizing
            return F.interpolate(
                img_tensor.unsqueeze(0), 
                size=(img_tensor.shape[1], 224, 224), 
                mode='trilinear', 
                align_corners=False
            ).squeeze(0)

        # Contrastive vs Supervised cases
        if isinstance(sample, tuple):
            processed_sample = tuple(process_tensor(s) for s in sample)
        else:
            processed_sample = process_tensor(sample)

        return {
            "img": processed_sample,
            "label": torch.tensor(label_int, dtype=torch.long),
            "label_name": label_enum.name.lower(),
            "fileset": row["FileSetId"],
            "idx": idx,
            "path": path
        }

    def get_labels(self) -> np.ndarray:
        return self.data["label_int"].values
    @property
    def labels(self) -> List[int]:
        """Instant access to labels without loading data."""
        return self.data["label_int"].tolist()

class NineClasses3DDatasetLoader:
    def __init__(self, 
                 batch_size: int = 32,
                 num_workers: int = 4,
                 pin_memory: bool = True,
                 image_size: tuple[int,int] = (224, 224),
                 base_dir: str | Path = PROCESSED_DATA_DIR / "nineclasses3D",
                 train_transform: Optional[Union[Callable, transforms.Compose]] = None,
                 test_transform: Optional[Union[Callable, transforms.Compose]] = None,
                 drop_last: bool = True,
                ):

        self.base_dir = Path(base_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.image_size = image_size
    
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.drop_last = drop_last
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def load_csvs(self):
        if not self.base_dir.exists():
            raise FileNotFoundError(f"Processed data directory not found at {self.base_dir}. "
                                    "Did you run the DatasetSaver yet?")

        # Helper to safely parse strings into tuples
        def parse_tuple(val):
            try:
                return ast.literal_eval(val) if isinstance(val, str) else val
            except (ValueError, SyntaxError):
                return val

        dfs = {}
        for split in DatasetSplit:
            file_path = self.base_dir / f"{split.value}.csv"
            df = pd.read_csv(file_path)
            
            # Deserialize shape columns from string to tuple
            if "shape" in df.columns:
                df["shape"] = df["shape"].apply(parse_tuple)
            if "shape_original" in df.columns:
                df["shape_original"] = df["shape_original"].apply(parse_tuple)
                
            dfs[split] = df

        return dfs[DatasetSplit.TRAIN], dfs[DatasetSplit.VAL], dfs[DatasetSplit.TEST]

    def build_datasets(self):
        train_df, val_df, test_df = self.load_csvs()

        self.train_dataset = FlexDataset3D(train_df, self.train_transform)
        self.val_dataset = FlexDataset3D(val_df, self.test_transform)
        self.test_dataset = FlexDataset3D(test_df, self.test_transform)

        return self.train_dataset, self.val_dataset, self.test_dataset

    def build_dataloaders(self, sampler=None, batch_sampler=None):
        if self.train_dataset is None:
            self.build_datasets()

        loader_args = {
            "dataset": self.train_dataset,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
        }
        if batch_sampler is not None:
            loader_args["batch_sampler"] = batch_sampler
        else:
            loader_args["batch_size"] = self.batch_size
            loader_args["sampler"] = sampler
            loader_args["shuffle"] = (sampler is None) # Shuffle if no custom sampler
            loader_args["drop_last"] = self.drop_last

        train_loader = DataLoader(**loader_args)


        val_loader = DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=self.pin_memory
        )

        test_loader = DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=self.pin_memory
        )

        return train_loader, val_loader, test_loader

    def build_stratified_subsets(self, samples_per_class: int = 2):
        """
        Creates stratified subsets for fast development.
        Ensures every class in NineClassesLabel is present in both train and val.
        """
        if self.train_dataset is None:
            self.build_datasets()

        def get_strat_indices(df: pd.DataFrame):
            strat_indices = []
            # Get stratified indixes
            for class_id in NineClassesLabel.class_ids():
                class_rows = df[df["label_int"] == class_id]
                if len(class_rows) > 0:
                    selected = class_rows.index[:min(len(class_rows), samples_per_class)] # Positional index
                    strat_indices.extend(df.index.get_indexer(selected).tolist())
                else:
                    print(f"Warning: Class {class_id} ({NineClassesLabel(class_id).name}) missing from split.")
            
            return strat_indices

        train_strat_idx = get_strat_indices(self.train_dataset.data)
        val_strat_idx = get_strat_indices(self.val_dataset.data)

        # Create Subsets
        self.train_dataset = torch.utils.data.Subset(self.train_dataset, train_strat_idx)
        self.val_dataset = torch.utils.data.Subset(self.val_dataset, val_strat_idx)

        return self.train_dataset, self.val_dataset
    

# BUILD DATASET
def run_test_builder(csv_path: str, max_samples_per_split: int = 5):
    """
    Build a small subset of the dataset for debugging purposes.
    This avoids processing the full dataset.
    """

    print("Running TEST dataset builder (small subset)...")

    saver = NineClasses3DDatasetSaver(csv_path=csv_path)

    # Load raw splits
    train_df, val_df, test_df, min_frames = saver._load_raw_data()

    # Subsample each split
    train_df = train_df.sample(min(max_samples_per_split, len(train_df)), random_state=42)
    val_df   = val_df.sample(min(max_samples_per_split, len(val_df)), random_state=42)
    test_df  = test_df.sample(min(max_samples_per_split, len(test_df)), random_state=42)

    # Override saver pipeline by monkey-patching splits
    splits = [train_df, val_df, test_df]

    base_save_path = saver._get_save_path() / "TEST_RUN"
    os.makedirs(base_save_path, exist_ok=True)

    for split_enum, df in zip(DatasetSplit, splits):
        save_path = base_save_path / split_enum.value
        os.makedirs(save_path, exist_ok=True)

        csv_name = base_save_path / split_enum.value
        saver._make_new_csv_file(csv_name)

        print(f"\nProcessing {split_enum.name} (TEST subset)...")

        for i, row in tqdm(df.iterrows(), total=len(df)):
            image_path = row["sample_path"]

            try:
                img, metadata = saver._get_image_3d(image_path, min_frames)

                img_name = f"{save_path}/volume_{i+1}.npz"

                row_out = row.copy()
                row_out["img"] = str(img_name)
                row_out["shape"] = metadata["shape"]
                row_out["shape_original"] = metadata["shape_original"]
                row_out["manufacturer"] = metadata["manufacturer"]

                saver._append_to_csv_file(csv_name, row_out)

                np.savez_compressed(img_name, img=img, **metadata)

            except Exception as e:
                print(f"Skipping {image_path} | Error: {e}")

    print("\nTEST dataset builder completed.")


def run_full_builder(csv_path: str):
    print("Running FULL dataset builder...")
    if not Path(csv_path).exists():
        print(f"CSV not found at {csv_path}. Attempting merge...")
        merge_full_dataset()
    saver = NineClasses3DDatasetSaver(csv_path=csv_path)
    if saver.check_preprocessed_data_saved():
        print("Dataset already exists. Skipping build.")
    else:
        saver.save_dataset()
        print("Full dataset creation complete.")

if __name__ == "__main__":
    print(f"Project Root: {PROJ_ROOT}")
    print(f"Target Directory: {DATA_DIR_OPTIMA3D}")

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="full", choices=["full", "test"])
    parser.add_argument("--csv", type=str, default=str(CSV_FILE))
    parser.add_argument("--test_samples", type=int, default=5)

    args = parser.parse_args()

    if args.mode == "test":
        run_test_builder(args.csv, max_samples_per_split=args.test_samples)
    else:
        run_full_builder(args.csv)