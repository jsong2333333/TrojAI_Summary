import numpy as np
import torch
from collections import OrderedDict

ORIGINAL_LEARNED_PARAM_DIR = './learned_parameters'
LAYERS_TO_MODEL_ARCH = {53: 'tinyroberta-squad2',
                        101: 'roberta-base-squad2',
                        558: 'mobilebert-uncased-squad-v2'}
MODEL_ARCH = ['tinyroberta-squad2', 'roberta-base-squad2', 'mobilebert-uncased-squad-v2']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _get_weight_features(model_repr, dim=(), normalize=False):
    weight_features = []
    for backbone_params in model_repr.values():
        if normalize:
            norm = torch.linalg.norm(backbone_params.reshape(backbone_params.shape[0], -1), ord=2)
            backbone_params  = backbone_params/norm
        weight_features += torch.amax(backbone_params, dim=dim).flatten().detach().cpu().tolist()
        weight_features += torch.mean(backbone_params, dim=dim).flatten().detach().cpu().tolist()
        end_dim = -1*(len(backbone_params.shape) - len(dim)) #- 1
        sub = torch.mean(backbone_params, dim=dim) - torch.median(torch.flatten(backbone_params, start_dim=0, end_dim=end_dim), dim=end_dim)[0]
        weight_features += sub.flatten().detach().cpu().tolist()
        weight_features += torch.median(torch.flatten(backbone_params, start_dim=0, end_dim=end_dim), dim=end_dim)[0].flatten().detach().cpu().tolist()
        weight_features += torch.sum(backbone_params, dim=dim).flatten().detach().cpu().tolist()
    return weight_features


def get_model_features(model_repr, infer=True): 
    features = []

    features.extend(_get_weight_features(model_repr, dim=(0,), normalize=True))

    if infer:
        return np.asarray([features])
    else:
        return features
