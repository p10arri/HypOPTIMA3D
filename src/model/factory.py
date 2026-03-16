import torch.nn as nn
from omegaconf import OmegaConf

from src.utils.enums import TrainingMode, Space
from src.model.transformer import Transformer
from src.model.baseline import BaselineTransformer
# from .wrappers import SiameseWrapper

def get_model(cfg, training_mode: TrainingMode) -> nn.Module:
    

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    params = cfg_dict['model']['encoder']['params']
    
    if 'space' in params and isinstance(params['space'], str):
        try:
            space_key = params['space'].upper()
            params['space'] = Space[space_key]
        except KeyError:
            raise ValueError(f"Space '{params['space']}' not found in Space Enum")

    if training_mode == TrainingMode.CLASSIFICATION:
        backbone = BaselineTransformer(**params)
    elif training_mode == TrainingMode.CONTRASTIVE:
        backbone = Transformer(**params)
    else:
        raise ValueError(f"Unsupported training mode: {training_mode}. ") 
    # TODO
    # # Wrap the model if using SimSiam (Self-Supervised)
    # if training_mode == TrainingMode.SIMSIAM:
    #     return SiameseWrapper(**params)

    return backbone
