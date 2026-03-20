import hydra
from omegaconf import DictConfig, OmegaConf
from loguru import logger
import torch
import geoopt

from src.data.dataset_builder import NineClassesDatasetLoader
from src.data.augmentations import get_augmentations
from src.data.sampler import SamplerFactory

from src.trainer import Trainer, seed_all
from src.model.vit3d import ViT3D
from src.model.projector import Projector
from src.losses import get_loss
from src.utils.enums import Space, TrainingMode, Augmentation

@hydra.main(config_path="config", config_name="config.yaml", version_base="1.3")
def main(cfg: DictConfig):
    logger.add("runtime.log", rotation="500 MB", backtrace=True, diagnose=True)
    seed_all(cfg.seed)
    
    # 1. Resolve Enums from Config
    space_type = Space(cfg.space.name)
    aug_mode = Augmentation(cfg.augmentations) # Based on your defaults: oct_classifier
    
    # Logic: If simsiam is in defaults/flags, it takes precedence over space training_mode
    if cfg.get("simsiam", False):
        training_mode = TrainingMode.SIMSIAM
    else:
        training_mode = TrainingMode(cfg.space.training_mode)

    logger.info(f"🚀 Starting Experiment: {cfg.experiment}")
    logger.info(f"Mode: {training_mode.name} | Space: {space_type.value}")

    # 2. Data Preparation
    train_aug, test_aug = get_augmentations(
        aug_mode=aug_mode.value, 
        transform_size=cfg.data.image_size,
        training_mode=training_mode
    )

    loader = NineClassesDatasetLoader(
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        train_transform=train_aug,
        test_transform=test_aug,
    )
    
    train_ds, val_ds, test_ds = loader.build_datasets()
    sampler, batch_sampler = SamplerFactory.get_sampler(cfg, train_ds)

    train_loader, val_loader, test_loader = loader.build_dataloaders(
        sampler=sampler,
        batch_sampler=batch_sampler
    )

    # 3. Model Assembly (Backbone + Projector Head)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    
    backbone = ViT3D(
        img_size=cfg.data.image_size,
        patch_size=cfg.model.patch_size,
        embed_dim=cfg.model.embed_dim,
        num_heads=cfg.model.num_heads,
        num_frames=16, # Or from cfg if dynamic
        skip_class_head=True # We want the features, not the ViT's internal head
    ).to(device)

    # Instantiate the Projector directly as per your setup
    head = Projector(
        space=space_type,
        embed_dim=backbone.out_dim,
        num_classes=cfg.data.num_classes,
        curvature=cfg.space.curvature,
        clip_r=cfg.space.get("clip_r", 1.0)
    ).to(device)

    # Wrap them into a single module for the trainer
    model = torch.nn.Sequential(torch.nn.ModuleDict({
        'backbone': backbone,
        'head': head
    }))

    # 4. Loss and Optimizer
    loss_fn = get_loss(cfg, training_mode=training_mode)

    # IMPORTANT: Use RiemannianAdam for non-Euclidean geometries
    # Standard Adam for Euclidean, RiemannianAdam for others.
    if space_type == Space.EUCLIDEAN:
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=cfg.space.lr, 
            weight_decay=cfg.optimizer.weight_decay
        )
    else:
        optimizer = geoopt.optim.RiemannianAdam(
            model.parameters(), 
            lr=cfg.space.lr, 
            weight_decay=cfg.optimizer.weight_decay
        )

    # 5. Execution
    trainer = Trainer(
        cfg=cfg,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer, # Pass the pre-built optimizer
        train_loader=train_loader,
        eval_loader=val_loader,
        training_mode=training_mode,
        space=space_type
    )

    logger.info(f"Starting training loop for {cfg.epochs} epochs...")
    trainer.fit() 

if __name__ == "__main__":
    main()