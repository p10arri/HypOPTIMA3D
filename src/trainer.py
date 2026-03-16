from __future__ import annotations
from typing import Dict, List, Union, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import OneCycleLR

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
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Set random seed to {seed}")


class Trainer:
    def __init__(self, 
                cfg: DictConfig, 
                model: nn.Module,
                loss_fn: nn.Module,
                train_loader: DataLoader, 
                eval_loader: DataLoader,               
                optimizer: Optimizer = None,
                scheduler: Optional[_LRScheduler] = None
                ):

        self.cfg = cfg
        seed_all(cfg.seed)
        self.device = self._setup_device()
        
        self.training_mode = TrainingMode(cfg.training_mode)
        self.space = Space(cfg.space.name)

        self.model = model.to(self.device)
        self.loss_fn = loss_fn
        
        self.optimizer = optimizer

        self.train_loader = train_loader
        self.eval_loader = eval_loader
        
        self.scheduler = scheduler

        self.evaluator = Evaluator(
            device=self.device, 
            hyp_c=self.cfg.space.curvature,
            k_list = [1, 2, 4, 8, 16, 32])

        # Training strategy
        self.simsiam = cfg.simsiam
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.early_stopping_patience = self.cfg.get("early_stopping_patience", None)
        self.early_stop_counter = 0
        self.monitor_mode = "max"
        self.best_score = -float('inf')
        self.training_stage = "init" # Track current training stage
        self.total_epochs_run = 0   # Track absolute progress across phases

        self.model_save_path: Path = Path(cfg.model_path)
        self.model_save_path.mkdir(parents=True, exist_ok=True)

        self._setup_wandb()
        logger.info(f"Trainer initialized for {self.training_mode.name}-{self.space.name}-{self.model.encoder_variant.upper()}")

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
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(self.device)
            elif isinstance(batch[key], list): # two augmented views
                #skip if its a list of strings
                batch[key] = [v.to(self.device) if isinstance(v, torch.Tensor) else v for v in batch[key]]
        return batch

    def adjust_learning_rate(self, epoch: int, max_epoch: int):        
        warmup_epochs = self.cfg.optimizer.warmup_epochs
        for param_group in self.optimizer.param_groups:
            # Save initial LR if not already saved
            init_lr = param_group.setdefault("initial_lr", param_group["lr"])
            fix_lr = param_group.get("fix_lr", False)

            # Linear warmup
            if epoch < warmup_epochs:
                lr = init_lr * (epoch + 1) / warmup_epochs
            else:
                # Cosine decay with floor
                min_lr = init_lr * 0.001
                decay_ratio = 0.5 * (1.0 + math.cos(math.pi * (epoch - warmup_epochs) / (max_epoch - warmup_epochs)))
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

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_score': self.best_score,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'epoch': self.current_epoch,
            'global_step': self.global_step,
        }

        latest_path = self.model_save_path / "latest_model.pt"
        best_path = self.model_save_path / "best_model.pt"
        torch.save(checkpoint, latest_path)
        if is_best:
            torch.save(checkpoint, self.model_save_path / 'best_model.pt')
        
        self.model.train()
        return is_best

    def load_checkpoint(self, checkpoint_path: Path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_score = checkpoint["best_score"]
        
        logger.info(f"Checkpoint loaded successfully! \n Resumed at epoch {self.current_epoch}")
    
    def _train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        self.model.train()
        batch = self._move_batch_to_device(batch)
        images = batch['img']
        targets = batch['label']
        self.optimizer.zero_grad()

        if self.training_mode == TrainingMode.CLASSIFICATION:
            # images is a single Tensor
            outputs = self.model(images)
            loss, stats = self.loss_fn(outputs, targets)
            
        elif self.training_mode == TrainingMode.CONTRASTIVE:
            # images is a list [view1, view2]
            z1 = self.model(images[0])
            z2 = self.model(images[1])
            loss, stats = self.loss_fn(z1, z2)
            outputs = z1
            
        elif self.training_mode == TrainingMode.SIMSIAM:
            # SimSiam logic (assumes model returns projections/predictions) #TODO: Not integrated
            p1, p2, z1, z2 = self.model(images[0], images[1])
            loss, stats = self.loss_fn(p1, p2, z1, z2)
            outputs = z1
        
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        self.optimizer.step()

        norm = torch.norm(outputs, dim=-1).mean().item()
        stats.update({"embedding_norm": norm, "grad_norm": grad_norm})
        return loss.item(), stats

    @torch.no_grad()
    def evaluate(self) -> float:
        self.model.eval()
        eval_metrics = {}
        total_loss, total_batches = 0.0, 0
        
        # For Classification Accuracy
        correct = 0
        total_samples = 0

        for batch in self.eval_loader:
            batch = self._move_batch_to_device(batch)
            images = batch['img']
            targets = batch['label']

            if self.training_mode == TrainingMode.CLASSIFICATION:
                outputs = self.model(images)
                loss, stats = self.loss_fn(outputs, targets)
                # Accuracy calc
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == targets).sum().item()
                total_samples += targets.size(0)
            else:
                # Use first view for Contrastive validation loss
                img_v = images[0] if isinstance(images, list) else images
                z1 = self.model(img_v)
                # Eval loss is a placeholder
                loss = torch.tensor(0.0).to(self.device) 
                stats = {}
            
            total_loss += loss.item()
            total_batches += 1

        avg_loss = total_loss / total_batches
        eval_metrics["eval/loss"] = avg_loss

        # Logic accuracy for checkpointing
        if self.training_mode == TrainingMode.CLASSIFICATION:
            score = correct / total_samples
            eval_metrics["eval/accuracy"] = score
        else:
            # SSL Metric Learning Recall
            recall_head, recall_body, _, _ = self.evaluator.evaluate_head_body(
                self.model, self.eval_loader
            )
            eval_metrics["eval/recall_head"] = recall_head
            eval_metrics["eval/recall_body"] = recall_body
            # Use the RecallHead@1 as score
            score = recall_head["recall@1"] if isinstance(recall_head, dict) else recall_head 

        # Log to wandb
        wandb_logs = {"global_step": self.global_step, "epoch": self.current_epoch}
        for k, v in eval_metrics.items():
            if isinstance(v, dict):
                for sub_k, sub_v in v.items(): wandb_logs[f"{k}/{sub_k}"] = sub_v
            else: wandb_logs[k] = v
        wandb.log(wandb_logs)

        logger.info(f"Epoch {self.current_epoch} | Score: {score:.4f} | Loss: {avg_loss:.4f}")
        is_best = self.save_checkpoint(score)
        self.model.train()
        return score, is_best

    def train(self, num_epochs: int, resume_from: bool = False):
        logger.disable("src.evaluator")       

        remaining_epochs = num_epochs - self.current_epoch

        logger.info(f"Training stage: {self.training_stage} for a remaining of {remaining_epochs} epochs")

        progress_bar = tqdm(total=num_epochs * len(self.train_loader), desc=f"Training[{self.training_stage}]", dynamic_ncols=True)

        if resume_from:
            checkpoint_path = self.model_save_path / self.training_stage / "latest_model.pt"
            logger.info(f"Resuming training from latest model of run: {self.cfg.experiment}")

            if checkpoint_path.exists():
                self.load_checkpoint(checkpoint_path)
            else:
                logger.warning("Checkpoint not found, starting fresh")
        else:
            logger.info("Starting fresh training (ignoring previous checkpoints)")

        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            epoch_loss = 0.0

            # Step sampler if using custom batch sampler
            if hasattr(self.train_loader.batch_sampler, "set_epoch"):
                self.train_loader.batch_sampler.set_epoch(epoch)

            if not self.scheduler:
                self.adjust_learning_rate(epoch, num_epochs)
                
            for batch in self.train_loader:
                loss, stats = self._train_step(batch)
                epoch_loss += loss
                self.global_step += 1

                wandb.log({
                    "train/loss": loss,
                    "train/stage_id": 1 if self.training_stage == "linear_probing" else 2,
                    "stage": self.training_stage,
                    **{f"train/{k}": v for k, v in stats.items()}
                }, step=self.global_step)
                
                progress_bar.set_description(f"Stage: {self.training_stage} | Epoch: {epoch+1} | Loss: {loss}")
                progress_bar.update(1)

            avg_train_loss = epoch_loss / len(self.train_loader)
            wandb.log({
                "epoch": epoch,
                "stage": self.training_stage,
                "train/epoch_loss": avg_train_loss
            }, step=self.global_step)


            if self.scheduler:
                self.scheduler.step()
                wandb.log({"train/lr": self.optimizer.param_groups[0]["lr"]}, step=self.global_step)
            
            # Evaluation checkpoint
            if (epoch + 1) % self.cfg.eval_frequency == 0 and epoch >= num_epochs//10:
                score, is_best = self.evaluate()
                if self.early_stopping_patience:
                    if is_best:
                        self.early_stop_counter = 0
                    else:
                        self.early_stop_counter += 1
                        if self.early_stop_counter >= self.early_stopping_patience:
                            logger.warning(f"Early stopping triggered at epoch {epoch}")
                            break
        progress_bar.close()
        # wandb.finish()
        logger.info("Training completed")
   
    def toggle_backbone_freeze(self, freeze: bool):
        # Set everything to NOT trainable
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Only unfreeze the head
        if freeze:
            for param in self.model.model.head.parameters():
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
        
    def fit(self, lr_lp:float, lr_ft:float):
        """Linear Probing then Fine-Tuning phases."""
        
        # --- LINEAR PROBING ---
        epochs_lp = self.cfg.epochs.linear_probing
        if epochs_lp > 0:
            logger.info(">>> STARTING PHASE 1: LINEAR PROBING")
            self.training_stage = "linear_probing"
            self.toggle_backbone_freeze(freeze=True)
            # Re-initialize optimizer for only trainable params
            self.optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()), 
                lr=lr_lp, 
                weight_decay=self.cfg.optimizer.weight_decay
            )
            self.train(num_epochs=epochs_lp)

        # --- FINE-TUNING ---
        logger.info(">>> STARTING PHASE 2: FINE-TUNING")
        self.training_stage = "finetuning"
        self.toggle_backbone_freeze(freeze=False)
        
        # Set a much lower learning rate for fine-tuning
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=lr_ft, 
            weight_decay=self.cfg.optimizer.weight_decay
        )
        
        # Continue training until the total epochs specified in cfg
        epochs_ft = self.cfg.epochs.finetuning
        total_epochs = epochs_lp + epochs_ft

        self.early_stop_counter = 0 # Restart the patience
        self.train(num_epochs=total_epochs)

        wandb.finish()

    def simple_fit(self, lr:float):
        """Standard end-to-end classification training."""
        logger.info(f">>> STARTING SIMPLE CLASSIFICATION TRAINING")
        self.training_stage = "standard_classification"
        
        # Ensure all parameters are trainable
        for param in self.model.parameters():
            param.requires_grad = True
            
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=lr, 
            weight_decay=self.cfg.optimizer.weight_decay
        )
        
        self.train(num_epochs=self.cfg.epochs.total)
        wandb.finish()