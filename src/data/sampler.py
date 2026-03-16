import torch
from torch.utils.data import Sampler
import numpy as np
import collections
import math
from typing import List, Dict, Optional, Sequence, Iterator, Any, Tuple
from loguru import logger

from src.utils.enums import ModelVariant, NineClassesLabel, Space

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
        Returns (sampler, batch_sampler) based on ModelVariant and SimSiam state.
        """
        sampler = None
        batch_sampler = None
        space = Space(cfg.space.name)

        if cfg.simsiam or space == Space.EUCLIDEAN:
            logger.info("Using standard PyTorch random sampling.")
            return None, None

        try:
            model_variant = ModelVariant(cfg.model.name)
        except ValueError:
            logger.error(f"Unknown model variant in config: {cfg.model.name}")
            raise

        labels = dataset.get_labels()
        seed = cfg.seed

        # Logic for ViT / DINO (Batch Sampler)
        if model_variant in [ModelVariant.VIT, ModelVariant.DINO]:
            logger.info(f"Assigning AllClassesKSampler for {model_variant.name}")
            
            num_classes = NineClassesLabel.num_classes()
            # Ensure at least 1 image per class in a batch
            k_val = max(1, cfg.data.batch_size // num_classes)
            
            batch_sampler = AllClassesKSampler(
                labels=labels, 
                k_per_class=k_val, 
                seed=seed
            )

        # Logic for MLP (Standard Sampler) # TODO:CURRENTLY NOT IN USE
        elif model_variant == ModelVariant.MLP:
            logger.info("Assigning UniqueSubsetSampler for MLP")
            
            sampler = UniqueSubsetSampler(
                labels=labels, 
                m_per_class=cfg.data.sampling.m_per_class, 
                seed=seed
            )

        return sampler, batch_sampler