from __future__ import annotations
from typing import Dict, List, Union, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
import monai
import numpy as np
import math
from loguru import logger
from pathlib import Path
import wandb

from src.utils.enums import TrainingMode, Space, NineClassesLabel
from src.evaluator import Evaluator

def seed_all(seed):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    monai.utils.set_determinism(seed=seed)
    torch.use_deterministic_algorithms(False) # Test for error
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Set random seed to {seed}")


class Trainer:    
    def __init__(
        self, 
        cfg: DictConfig,
        model: nn.Module,
        optimizer: Optimizer,
        loss_fn: nn.Module,
        train_loader: DataLoader, 
        eval_loader: DataLoader,
        scheduler: Optional[_LRScheduler] = None,
        stage_name: str = "",
    ):

        self.cfg = cfg
        seed_all(cfg.get("seed", 42))
        self.device = self._setup_device()
        
        self.training_mode = TrainingMode(cfg.training_mode)
        self.space = Space(cfg.space.get("name", "euclidean"))
        self.curvature = OmegaConf.select(cfg, "space.curvature", default=0.0)

        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.eval_loader = eval_loader

        self.evaluator = Evaluator(
            device=self.device, 
            space = self.space,
            training_mode=self.training_mode,
            hyp_c=self.curvature,
            k_list=cfg.get("eval_k_list", [1, 2, 4, 8, 16, 32])
        )

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.monitor_mode = "max" # Accuracy and Recall are always "max"
        self.best_score = -float('inf')
        self.early_stop_counter = 0
        self.early_stopping_patience = cfg.trainer.early_stopping_patience

        # Paths
        self.resume_from = cfg.trainer.resume_from
        base_path = Path(cfg.model_path)
        if stage_name:
            self.cfg.wandb.run_name = self.cfg.wandb.run_name + f"_{stage_name}"
            self.model_save_path = base_path / stage_name
        else:
            self.model_save_path = base_path
        
        self.model_save_path.mkdir(parents=True, exist_ok=True)
        # # Dont store models if fast_dev_run enabled # Store weitgths for finetuning
        # if not cfg.get("fast_dev_run", False):
        #     self.model_save_path.mkdir(parents=True, exist_ok=True)
        # else:
        #     logger.info("Fast dev run: Model saving directory creation skipped.")

        self._setup_wandb()
        logger.info(
            f"Trainer Initialized | Mode: {self.training_mode.name} | "
            f"Space: {self.space.name} (c={self.curvature}) | "
            f"Model: {self.model.__class__.__name__}"
        )

        self.wandb = False if cfg.wandb.mode== "disabled" else True
        if self.wandb:
            self._setup_wandb()
        logger.info(f"Trainer initialized for {self.training_mode.name}-{self.space.name}-{self.model.__class__.__name__.upper()}")

    def _setup_device(self):
        return torch.device(self.cfg.device)

    def _setup_wandb(self):
        wandb.init(
            project=self.cfg.wandb.project,
            entity=self.cfg.wandb.entity,
            name=self.cfg.wandb.run_name,
            mode=self.cfg.wandb.mode,
            notes=self.cfg.wandb.notes,
            resume=self.cfg.wandb.resume,
            config=OmegaConf.to_container(
                self.cfg,
                resolve=True,
                # throw_on_missing=True,
            ),
            
        ) 
        wandb.define_metric("epoch")
        wandb.define_metric("train/*", step_metric="global_step")
        wandb.define_metric("eval/*", step_metric="epoch")


    def _move_batch_to_device(self, batch: Dict[str, torch.Tensor]):
        """Move a batch (dict, list, or tensor) to the target device."""
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        elif isinstance(batch, dict):
            return {k: self._move_batch_to_device(v) for k, v in batch.items()}
        
        elif isinstance(batch, (list, tuple)): #two_views
            return [self._move_batch_to_device(v) for v in batch]
        
        return batch

    def adjust_learning_rate(self, epoch: int, max_epoch: int): 
        warmup_epochs = self.cfg.get("warmup_epochs", 10)
        init_lr = self.cfg.optimizer.lr

        for param_group in self.optimizer.param_groups:
            # Handle frozen backbone (fixed lr)
            if param_group.get("fix_lr", False):
                continue

            # Save initial LR if not already saved
            init_lr = param_group.setdefault("initial_lr", param_group["lr"])
            fix_lr = param_group.get("fix_lr", False)

            # Linear warmup
            if epoch < warmup_epochs:
                lr = init_lr * (epoch + 1) / warmup_epochs
            else:
                # Cosine decay with floor
                min_lr = init_lr * 0.001
                decay_ratio = 0.5 * (1.0 + math.cos(math.pi * (epoch - warmup_epochs) / max(1, max_epoch - warmup_epochs)))
                lr = min_lr + (init_lr - min_lr) * decay_ratio

            param_group["lr"] = init_lr if fix_lr else lr


    def save_checkpoint(self, current_score: float):
        self.model.eval()
 
        if self.monitor_mode == "max":
            is_best = current_score > self.best_score
        else:
            is_best = current_score < self.best_score

        if is_best:
            self.best_score = current_score
            logger.info(f"New best model found! Score: {current_score:.4f}")

        # if self.cfg.get("fast_dev_run", False):
        #     logger.debug("Fast dev run enabled: Skipping checkpoint file generation.")
        #     return is_best
        
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'best_score': self.best_score,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'config': OmegaConf.to_container(self.cfg, resolve=True)
        }

        latest_path = self.model_save_path / "latest_model.pt"
        best_path = self.model_save_path / "best_model.pt"
        torch.save(checkpoint, latest_path)
        if is_best:
            torch.save(checkpoint, best_path)
        
        self.model.train()
        return is_best

    def load_checkpoint(self, checkpoint_path: Path):
        if not checkpoint_path.exists():
            logger.error(f"No checkpoint found at {checkpoint_path}")
            return
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Restore counter
        self.current_epoch = checkpoint["epoch"] +1 # resume from next epoch
        self.global_step = checkpoint["global_step"]
        self.best_score = checkpoint["best_score"]
        
        logger.info(f"Checkpoint loaded successfully! \n Resumed from epoch {self.current_epoch}"
                    f"(Previous Best Score: {self.best_score:.4f})")
    
    def _train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        self.model.train()
        batch = self._move_batch_to_device(batch)

        # Unpack batch
        if isinstance(batch, (list, tuple)):
            # Standard PyTorch/MedMNIST: [data, labels]
            images, targets = batch[0], batch[1]
        else:
            # OPTIMA3D format: {"img": data, "label": labels}
            images = batch['img']
            targets = batch['label']

        self.optimizer.zero_grad()

        if self.training_mode == TrainingMode.SUPERVISED:
            # images is a single Tensor
            outputs = self.model(images)['logits']
            loss, stats = self.loss_fn(outputs, targets)
            
        elif self.training_mode == TrainingMode.CONTRASTIVE:
            # images are two views of same image
            z1 = self.model(images[0])['embeddings']
            z2 = self.model(images[1])['embeddings']
            loss, stats = self.loss_fn(z1, z2)
            outputs = z1
            
        elif self.training_mode == TrainingMode.SIMSIAM:
            # SimSiam logic (assumes model returns projections/predictions) #TODO: Not integrated
            p1, p2, z1, z2 = self.model(images[0], images[1])
            loss, stats = self.loss_fn(p1, p2, z1, z2)
            outputs = z1
        
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.get("grad_clip", 1.0))
        self.optimizer.step()

        with torch.no_grad():
            emb_norm = torch.norm(outputs, dim=-1).mean().item()
        
        stats.update({
            "embedding_norm": emb_norm, 
            "grad_norm": grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm
        })

        return loss.item(), stats

    @torch.no_grad()
    def evaluate(self) -> Tuple[float, bool]:

        eval_metrics = self.evaluator.run(
            model=self.model, 
            loader=self.eval_loader, 
            loss_fn=self.loss_fn
        )

        # Determine the primary score for checkpointing
        # Supervised -> Accuracy | Contrastive -> Recall@1
        if self.training_mode == TrainingMode.SUPERVISED:
            score = eval_metrics.get("eval/accuracy", 0.0)
        else:
            score = eval_metrics.get("eval/recall@1", 0.0)
    

        # wandb log
        if self.wandb:
            wandb_logs = {
            "epoch": self.current_epoch, 
            "global_step": self.global_step
            }

            for k, v in list(eval_metrics.items()):
                # Log Confusion Matrix
                if "confusion_matrix" in k:
                    wandb_logs["eval/conf_mat"] = wandb.plot.confusion_matrix(
                        probs=None,
                        y_true = eval_metrics.pop("raw_y_true"), # Remove from dict so it's not logged as raw data
                        preds = eval_metrics.pop("raw_y_pred"),
                        class_names = NineClassesLabel.class_names()
                    )
                else:
                    wandb_logs[k] = v

            wandb.log(wandb_logs)

        # 4. Logging and Checkpointing
        logger.info(f"Epoch {self.current_epoch} | Eval Score ({self.training_mode.name}): {score:.4f}")
        
        is_best = self.save_checkpoint(score)
        
        # Ensure model returns to training mode
        self.model.train()
        
        return score, is_best
    
    def train(self, num_epochs: int):
        if self.resume_from:
            checkpoint_path = self.model_save_path / "latest_model.pt"
            logger.info(f"Resuming training from latest model of run: {self.cfg.experiment}")

            if checkpoint_path.exists():
                self.load_checkpoint(checkpoint_path)
            else:
                logger.warning("Checkpoint not found, starting fresh")


        start_epoch = self.current_epoch
        total_iters = (num_epochs - start_epoch) * len(self.train_loader)

        progress_bar = tqdm(
            total=total_iters, 
            desc=f"[{self.training_stage.upper()}]", 
            dynamic_ncols=True
        )

        for epoch in range(start_epoch, num_epochs):
            self.current_epoch = epoch # Sync state
            epoch_loss = 0.0

            # Sync sampler
            if hasattr(self.train_loader, "sampler") and hasattr(self.train_loader.sampler, "set_epoch"):
                self.train_loader.sampler.set_epoch(epoch)
            elif hasattr(self.train_loader.batch_sampler, "set_epoch"):
                self.train_loader.batch_sampler.set_epoch(epoch)

            # Lr scheduling
            if not self.scheduler:
                self.adjust_learning_rate(epoch, num_epochs)           
            lr = self.optimizer.param_groups[0]["lr"]

            # Train step
            for batch in self.train_loader:
                loss, stats = self._train_step(batch)
                epoch_loss += loss
                self.global_step += 1

                if self.wandb:
                    wandb.log({
                        "train/loss": loss,
                        "train/lr": lr,
                        **{f"train/{k}": v for k, v in stats.items()}
                    }, step=self.global_step)
                
                progress_bar.set_postfix({"loss": f"{loss:.4f}", "lr": f"{lr:.6f}"})
                progress_bar.update(1)

            # Metrics
            avg_train_loss = epoch_loss / len(self.train_loader)
            if self.wandb: 
                wandb.log({"train/epoch_loss": avg_train_loss, "epoch": epoch, "stage": self.training_stage}, step=self.global_step)

            if self.scheduler:
                self.scheduler.step()
                if self.wandb:
                    wandb.log({"train/lr": self.optimizer.param_groups[0]["lr"], "epoch": epoch, "stage": self.training_stage}, step=self.global_step)

            # Evaluation and checkpoint
            val_interval = self.cfg.trainer.eval_frequency

            if (epoch + 1) % val_interval == 0:
                score, is_best = self.evaluate()
                # Early Stopping Logic
                if self.early_stopping_patience:
                    if is_best:
                        self.early_stop_counter = 0
                    else:
                        self.early_stop_counter += 1
                        if self.early_stop_counter >= self.early_stopping_patience:
                            logger.warning(f"Early stopping triggered at epoch {epoch}")
                            break
        
        progress_bar.close()
        logger.info(f"Training Stage {self.training_stage} Completed.")

    def toggle_backbone_freeze(self, freeze: bool):
        # Freeze everything
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Only unfreeze the head
        if freeze:
            for param in self.model.head.parameters():
                param.requires_grad = True
            logger.info("Backbone is FROZEN. Only Head is training.")
        else:
            # Fine-Tuning: Everything is trainable
            for param in self.model.parameters():
                param.requires_grad = True
            logger.info("Backbone is UNFROZEN.")

        # Debug trainable layers
        trainable_params = [name for name, p in self.model.named_parameters() if p.requires_grad]
        logger.info(f"Actual Trainable Params: {len(trainable_params)}")
        for name in trainable_params:
            logger.debug(f"Training: {name}")
        

    def fit(self, num_epochs: int):
        """Standard end-to-end classification training."""
        logger.info(f">>> STARTING STANDARD TRAINING")
        self.training_stage = "full_training"
        
        # Ensure all parameters are trainable
        for param in self.model.parameters():
            param.requires_grad = True
            
        self.train(num_epochs)
        if self.wandb:
            wandb.finish()


