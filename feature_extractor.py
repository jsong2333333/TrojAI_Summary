import numpy as np
import torch
from itertools import product


ORIGINAL_LEARNED_PARAM_DIR = './learned_parameters'
MODEL_ARCH = [''] #['SimplifiedRLStarter', 'BasicFCModel']


def _get_weight_features(model_repr, dim=(), normalize=False):
    weight_features = []
    weight_lens = []
    for backbone_params in model_repr.values():
        # pshape = len(param.shape)
        # if axis is None:
            # axis = tuple(range(-1, -1*(pshape), -1))
        # weight_features += np.max(param, axis= axis, keepdims=True).flatten().tolist()
        # weight_features += np.mean(param, axis= axis, keepdims=True).flatten().tolist()
        # sub = np.mean(param, axis= axis, keepdims=True) - np.median(param, axis= axis, keepdims=True)
        # weight_features += sub.flatten().tolist()
        # weight_features += np.median(param, axis= axis, keepdims=True).flatten().tolist()
        # weight_features += np.sum(np.abs(param), axis= axis, keepdims=True).flatten().tolist()
        if normalize:
            norm = torch.linalg.norm(backbone_params.reshape(backbone_params.shape[0], -1), ord=2)
            backbone_params  = backbone_params/norm
        weight_features += torch.amax(backbone_params, dim=dim).flatten().detach().cpu().tolist()
        weight_features += torch.mean(backbone_params, dim=dim).flatten().detach().cpu().tolist()
        end_dim = -1*(len(backbone_params.shape) - len(dim))
        sub = torch.mean(backbone_params, dim=dim) - torch.median(torch.flatten(backbone_params, start_dim=0, end_dim=end_dim), dim=end_dim)[0]
        weight_features += sub.flatten().detach().cpu().tolist()
        weight_features += torch.median(torch.flatten(backbone_params, start_dim=0, end_dim=end_dim), dim=end_dim)[0].flatten().detach().cpu().tolist()
        weight_features += torch.sum(backbone_params, dim=dim).flatten().detach().cpu().tolist()
        weight_lens.append(len(weight_features))
    return weight_features, weight_lens

def _get_eigen_features(model_repr, normalize=False, ssv_portion = 0.5):
    min_shape, params = 1, []
    for param in model_repr.values():
        if len(param.shape) > min_shape:
            if normalize:
                norm = torch.linalg.norm(param.reshape(param.shape[0], -1), ord=2)
                param  = param/norm
            reshaped_param = param.reshape(param.shape[0], -1)
            _, singular_values, _ = torch.linalg.svd(reshaped_param, False)
            ssv = torch.square(singular_values).flatten()
            params.extend(ssv.tolist()[:int(len(ssv)*ssv_portion)])
    return params


def keys_for_extraction(model):
    layer_names = ['actor', 'critic', 'state_emb']
    keys_to_dict = {} #{layer_name:1e8 for layer_name in layer_names}
    
    for k in model.keys():
        splitted_k = k.split('.')
        layer_name = splitted_k[0]
        if layer_name in layer_names:
            keys_to_dict.setdefault(layer_name, 1e8)
            n_layer = keys_to_dict[layer_name]
            ini_layer = min(n_layer, int(splitted_k[1]))
            keys_to_dict[layer_name] = ini_layer
            
    keys_for_extract = sorted([f'{k}.{v}.{wb}' for (k, v),  wb in product(keys_to_dict.items(), ['weight', 'bias'])])
    
    return {k:v for k, v in model.items() if k in keys_for_extract}


def get_model_features(model_repr, infer=True):
    
    # model_repr = keys_for_extraction(model_repr)
     
    features = []
    features += _get_weight_features(model_repr, normalize=True)[0]
    features += _get_eigen_features(model_repr, normalize=True, ssv_portion=0.4)

    if infer:
        return np.asarray([features])
    else:
        return features
