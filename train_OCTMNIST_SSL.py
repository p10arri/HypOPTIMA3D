import os
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra import compose, initialize

from omegaconf import DictConfig, OmegaConf
from loguru import logger
import torch
from torch.utils.data import DataLoader
import geoopt

from medmnist import INFO, OCTMNIST

from src.data.dataset_builder import NineClasses3DDatasetLoader
from src.data.augmentations import get_augmentations, get_augmentations2D
from src.data.sampler import SamplerFactory
from pathlib import Path

from src.trainer import Trainer, seed_all
from src.model.builder import get_model, get_vit3d
from src.losses import get_loss
from src.utils.enums import Space, TrainingMode, NineClassesLabel


@hydra.main(config_path="config", config_name="config.yaml", version_base="1.3")
def main(cfg: DictConfig):
    output_dir = HydraConfig.get().runtime.output_dir
    log_path = os.path.join(output_dir, "runtime.log")
    
    logger.add(log_path, rotation="500 MB", backtrace=True, diagnose=True)
    seed_all(cfg.seed)
    
    # Resolve Space and Training Mode
    space_type = Space(cfg.space.name)
    logger.info(f"🚀 PRETRAINING (OCTMNIST) + FINETUNING (OPTIMA) \n Experiment: {cfg.experiment}| Space: {space_type.value}")
    
    ## --------------------------------------------
    ## ----------- PRETRAINING STAGE --------------

    logger.info(f"Pretraining Stage...")
    # Override config parameters
    cfg.training_mode = "contrastive"
    cfg.model.pretrained = True
    stage_name = "pretrain"
    cfg.optimizer.lr = 1e-4 # High enough to pull/push embeddings significantly.
    cfg.optimizer.weight_decay = 0.05 # High weight decay prevents ViT from over-relying on single pixels
    cfg.optimizer.warmup_epochs = 10 
    cfg.trainer.epochs = 70
    cfg.trainer.early_stopping_patience = 15
    
    

    # TRAINING MODE
    training_mode = TrainingMode(cfg.training_mode)

    # DATA & AUGMENTATION
    # Augmentations work at slice level. E.g.: Rotations in 2D plane, around depth axis (slice rotation is propagated to all slices within the volume)
    train_aug, test_aug = get_augmentations2D(
        aug_mode=cfg.augmentations.name, 
        transform_size=cfg.data.image_size,
        training_mode=training_mode
    )
    
    train_ds = OCTMNIST(split='train', transform=train_aug, download=True) # OCTMNIST is 1x28x28 befor augmentations
    val_ds = OCTMNIST(split='val', transform=test_aug, download=True)

    # Get OCTMNIST Metadata
    sample_data = train_ds[0]
    sample_img = sample_data[0][0] if isinstance(sample_data[0], (list, tuple)) else sample_data[0]
    
    in_channels = sample_img.shape[0]
    num_frames = sample_img.shape[1] if len(sample_img.shape) > 2 else 1
    num_classes = len(INFO['octmnist']['label'])
    logger.info(f"Found {num_classes} unique classes in OCTMNIST: {INFO['octmnist']['label']}")

    if cfg.get("fast_dev_run", False):
        logger.warning("⚠️ FAST DEV RUN: Subsetting OCTMNIST")
        from sklearn.model_selection import train_test_split
        import numpy as np

        # Ensure we have the full label array flattened
        train_labels = np.array(train_ds.labels).flatten()
        val_labels = np.array(val_ds.labels).flatten()

        # Check: do we actually see 8 classes in the raw labels?
        unique_train = np.unique(train_labels)
        unique_val = np.unique(val_labels)
        logger.info(f"Found {len(unique_train)} unique classes in raw train_ds: {unique_train}")
        logger.info(f"Found {len(unique_val)} unique classes in raw val_ds: {unique_val}")

        # Stratified split for Train
        train_indices, _ = train_test_split(
            np.arange(len(train_labels)),
            train_size=num_classes * 10,
            stratify=train_labels,
            random_state=42
        )
        
        # Stratified split for Val
        val_indices, _ = train_test_split(
            np.arange(len(val_labels)),
            train_size=num_classes* 10,
            stratify=val_labels,
            random_state=42
        )

        train_ds = torch.utils.data.Subset(train_ds, train_indices)
        val_ds = torch.utils.data.Subset(val_ds, val_indices)

        epochs = 5
        cfg.wandb.mode = "disabled"
    else:
        epochs = cfg.trainer.epochs

    # Sampler
    sampler, batch_sampler = SamplerFactory.get_sampler(cfg, train_ds)

    # Loader
    train_loader = DataLoader(
        dataset=train_ds,
        sampler=sampler,
        batch_sampler=batch_sampler,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory
        )
    
    val_loader = DataLoader(
        dataset=val_ds,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory
    )

    # MODEL, LOSS & OPTIMIZER    
    model = get_model(cfg, num_classes, num_frames, in_channels)
    loss_fn = get_loss(cfg, training_mode=training_mode)
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=cfg.optimizer.lr, 
        weight_decay=cfg.optimizer.weight_decay
    )

    # PRETRAINING
    trainer = Trainer(
        cfg=cfg,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        train_loader=train_loader,
        eval_loader=val_loader,
        stage_name=stage_name
    )

    trainer.fit(epochs)

    logger.success("Pretraining stage completed")


    ## --------------------------------------------
    ## ----------- FINE-TUNING STAGE --------------
    logger.info("Transitioning to Fine-tuning Stage (OPTIMA 3D)...")

    # Override SPACE to Euclidean for Supervise learning:
    with initialize(config_path="config", version_base="1.3"):
        new_space_cfg = compose(config_name="config", overrides=["space=euclidean"])
        
        # 2. Update main cfg with the new space value
        # This replaces the entire 'space' block (name, thau, clip_radius)
        cfg.space = new_space_cfg.space
    # Override config parameters
    cfg.training_mode = "supervised"
    cfg.model.pretrained = True
    cfg.model.skip_class_head = False
    cfg.model.checkpoint_path = Path(cfg.model_path) / stage_name / "best_model.pt"
    stage_name = "finetune"
    cfg.optimizer.lr = 2e-5 # High enough to pull/push embeddings significantly.
    cfg.optimizer.weight_decay = 0.05 # High weight decay prevents ViT from over-relying on single pixels
    cfg.optimizer.warmup_epochs = 10 
    cfg.trainer.epochs = 200
    cfg.trainer.early_stopping_patience = 30

    # TRAINING MODE
    training_mode = TrainingMode(cfg.training_mode)

    # 3D Data & Augmentations
    train_aug_3d, test_aug_3d = get_augmentations(
        aug_mode=cfg.augmentations.name, 
        transform_size=cfg.data.image_size,
        training_mode=training_mode
    )

    loader = NineClasses3DDatasetLoader(
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        train_transform=train_aug_3d,
        test_transform=test_aug_3d,
    )
    
    train_ds, val_ds, _ = loader.build_datasets()

    if cfg.get("fast_dev_run", False):
        logger.warning("⚠️ FAST DEV RUN: Subsetting 3D Dataset")
        train_ds, val_ds = loader.build_stratified_subsets(samples_per_class=2)
        epochs = 5
    else:
        epochs = cfg.trainer.epochs

    # Get number of frames and input channels from dataset
    num_frames = val_ds[0]["img"].shape[1]
    in_channels = val_ds[0]["img"].shape[0]
    num_classes = NineClassesLabel.num_classes()

    # Sampler
    sampler, batch_sampler = SamplerFactory.get_sampler(cfg, train_ds)

    # Loader
    train_loader, val_loader, _ = loader.build_dataloaders(
        sampler=sampler,
        batch_sampler=batch_sampler
    )

    # Model for Finetuning
    # The model is the ViT3D with the classification head
    model = get_vit3d(cfg, num_classes, num_frames, in_channels)
    
    # Loss and Optimizer
    loss_fn = get_loss(cfg, training_mode=training_mode)
    optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=cfg.optimizer.lr, 
            weight_decay=cfg.optimizer.weight_decay
        )

    # Training
    trainer = Trainer(
        cfg=cfg,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        train_loader=train_loader,
        eval_loader=val_loader,
        stage_name=stage_name

    )
    trainer.fit(epochs)


    logger.success("✅ Full Pipeline (Pretrain + Finetune) Completed!")

if __name__ == "__main__":
    main()