from enum import Enum
from typing import List, Union, Any
import torch
import numpy as np

class Space(Enum):
    EUCLIDEAN = "euclidean"
    SPHERICAL = "spherical"
    HYPERBOLIC = "hyperbolic"

class TrainingMode(Enum):
    # Standard classification: 1 image -> logits -> CrossEntropy
    CLASSIFICATION = "classification" 
    # Self-supervised/Metric: 2 views -> embeddings -> PairwiseCELoss
    CONTRASTIVE = "contrastive"       
    # Siamese SSL: 2 views -> projection/prediction head -> Cosine Similarity
    SIMSIAM = "simsiam"

class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"

class NineClassesLabel(Enum):
    CNV1 = 0
    CNV2 = 1
    CNV3 = 2
    DME = 3
    GA = 4
    HEALTHY = 5
    IAMD = 6
    RVO = 7
    STARGARDT = 8

    @classmethod
    def num_classes(cls) -> int:
        return len(cls)
    
    @classmethod
    def class_names(cls) -> List[str]:
        return [c.name.lower() for c in cls]

    @classmethod
    def class_ids(cls) -> List[int]:
        return [c.value for c in cls]

    @classmethod
    def to_name(
        cls,
        labels: Union[int, List[int], torch.Tensor, np.ndarray]
    ) -> List[str]:
        # Professional handling of different input types
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy().flatten().tolist()
        elif isinstance(labels, np.ndarray):
            labels = labels.flatten().tolist()
        elif isinstance(labels, int):
            labels = [labels]
        
        # Maps the integer value back to the Enum name
        return [cls(label).name.lower() for label in labels]

    @classmethod
    def to_id(cls, names: Union[str, List[str]]) -> List[int]:
        if isinstance(names, str):
            names = [names]
        return [cls[name.upper()].value for name in names]

class Augmentation(Enum):
    NORMAL = "normal"
    HEAVY = "heavy"
    OCT_CLASSIFIER = "oct_classifier"

class OptimizerType(Enum):
    ADAM = "adam"
    SGD = "sgd"
    ADAMW = "adamw"

# Helper Functions
def to_space(value: str) -> Space:
    return Space(value)

def to_optimizer(value: str) -> OptimizerType:
    return OptimizerType(value)

def to_augmentation(value: str) -> Augmentation:
    return Augmentation(value)