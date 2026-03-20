import torch
from torch.utils.data import Sampler
import numpy as np
import collections
import math
from typing import List, Dict, Optional, Sequence, Iterator, Any, Tuple
from loguru import logger

from src.utils.enums import TrainingMode, NineClassesLabel, Space

def get_labels_to_indices(labels: Sequence) -> Dict[Any, np.ndarray]:
    """Maps each unique label to an array of indices for that label."""
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    
    labels_to_indices = collections.defaultdict(list)
    for i, label in enumerate(labels):
        labels_to_indices[label].append(i)
    
    return {k: np.array(v, dtype=int) for k, v in labels_to_indices.items()}

class BaseHypSampler(Sampler):
    def __init__(self, labels: Sequence, seed: int = 0):
        super().__init__(None)
        self.labels_to_indices = get_labels_to_indices(labels)
        self.all_classes = sorted(list(self.labels_to_indices.keys()))
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _safe_random_choice(self, idxs: np.ndarray, size: int, rng: np.random.Generator) -> List[int]:
        """Samples indices, allowing replacement if the class size is smaller than requested."""
        replace = len(idxs) < size
        return rng.choice(idxs, size=size, replace=replace).tolist()

class AllClassesKSampler(BaseHypSampler):
    """
    Every batch contains all P classes with K images per class.
    Effective batch_size = len(all_classes) * k_per_class.
    """
    def __init__(self, labels: Sequence, k_per_class: int, batches_per_epoch: Optional[int] = None, seed: int = 0):
        super().__init__(labels, seed)
        self.K = k_per_class
        self.P = len(self.all_classes)
        self.batch_size = self.P * self.K

        if batches_per_epoch is None:
            self.batches_per_epoch = math.ceil(len(labels) / self.batch_size)
        else:
            self.batches_per_epoch = batches_per_epoch

    def __len__(self) -> int:
        return self.batches_per_epoch

    def __iter__(self) -> Iterator[List[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        for _ in range(self.batches_per_epoch):
            batch_indices = []
            for cls in self.all_classes:
                idxs = self.labels_to_indices[cls]
                batch_indices.extend(self._safe_random_choice(idxs, self.K, rng))
            rng.shuffle(batch_indices)
            yield batch_indices

class UniqueSubsetSampler(BaseHypSampler):
    """
    Samples m indices per each unique label.
    """
    def __init__(self, labels: Sequence, m_per_class: int, seed: int = 0):
        super().__init__(labels, seed)
        self.m_per_class = m_per_class

    def __len__(self) -> int:
        return len(self.all_classes) * self.m_per_class

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng(self.seed + self.epoch)
        
        # Shuffle class order
        shuffled_classes = rng.permutation(self.all_classes)
        
        all_indices = []
        for cls in shuffled_classes:
            idxs = self.labels_to_indices[cls]
            all_indices.extend(self._safe_random_choice(idxs, self.m_per_class, rng))
            
        return iter(all_indices)


class SamplerFactory:
    
    @staticmethod
    def get_sampler(cfg, dataset: Any) -> Tuple[Optional[Sampler], Optional[Sampler]]:
        """
        Returns (sampler, batch_sampler) based on TrainingMode
        """
        sampler = None
        batch_sampler = None
        training_mode = TrainingMode(cfg.training_mode)

        if training_mode == TrainingMode.CONTRASTIVE:
            logger.info(f"Assigning AllClassesKSampler for {training_mode.name} mode")
            
            k_val = cfg.data.sampling.m_per_class
            
            labels = dataset.get_labels()
            
            batch_sampler = AllClassesKSampler(
                labels=labels, 
                k_per_class=k_val, 
                seed=cfg.seed
            )
            # Log the effective batch size to avoid OOM surprises
            effective_bs = batch_sampler.batch_size
            if effective_bs != cfg.data.batch_size:
                logger.warning(f"Sampler overriding config batch_size! New batch_size: {effective_bs} ({NineClassesLabel.num_classes()} classes * {k_val} per class)")

        else:
            # Standard SUPERVISED Euclidean training
            logger.info("Using standard PyTorch random sampling (None, None)")
            return None, None

        return sampler, batch_sampler