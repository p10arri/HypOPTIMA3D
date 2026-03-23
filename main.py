import os
import hydra
from hydra.core.hydra_config import HydraConfig

from omegaconf import DictConfig, OmegaConf
from loguru import logger
import torch
import geoopt

from src.data.dataset_builder import NineClasses3DDatasetLoader
from src.data.augmentations import get_augmentations
from src.data.sampler import SamplerFactory

from src.trainer import Trainer, seed_all
from src.model.builder import get_model
from src.losses import get_loss
from src.utils.enums import Space, TrainingMode, Augmentation, NineClassesLabel

@hydra.main(config_path="config", config_name="config.yaml", version_base="1.3")
def main(cfg: DictConfig):
    output_dir = HydraConfig.get().runtime.output_dir
    log_path = os.path.join(output_dir, "runtime.log")
    logger.add(log_path, rotation="500 MB", backtrace=True, diagnose=True)
    seed_all(cfg.seed)
    
    # Resolve Space and Training Mode
    space_type = Space(cfg.space.name)
    training_mode = TrainingMode(cfg.training_mode)

    logger.info(f"🚀 Experiment: {cfg.experiment} | Mode: {training_mode.name} | Space: {space_type.value}")

    # Data & Augmentation setup
    # Augmentations work at slice level. E.g.: Rotations in 2D plane, around depth axis (slice rotation is propagated to all slices within the volume)
    train_aug, test_aug = get_augmentations(
        aug_mode=cfg.augmentations.name, 
        transform_size=cfg.data.image_size,
        training_mode=training_mode
    )

    loader = NineClasses3DDatasetLoader(
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        train_transform=train_aug,
        test_transform=test_aug,
    )
    
    train_ds, val_ds, _ = loader.build_datasets()

    if cfg.get("fast_dev_run", False):
        logger.warning("⚠️ FAST DEV RUN: Using stratified subset (Instant)")
        # Override config
        cfg.wandb.mode = "disabled"
        cfg.results_path = "results/dev_trash"
        cfg.model_path = "results/dev_trash/models"
        cfg.logs_path = "results/dev_trash/logs"
        
        # Create subset
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

    train_loader, val_loader, _ = loader.build_dataloaders(
        sampler=sampler,
        batch_sampler=batch_sampler
    )

    # Model 
    model = get_model(cfg, num_classes, num_frames, in_channels)

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
    )

    trainer.fit(epochs)

if __name__ == "__main__":
    main()