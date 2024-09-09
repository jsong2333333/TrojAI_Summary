import numpy as np

ORIGINAL_LEARNED_PARAM_DIR = './learned_parameters'
MODEL_ARCH = ['SimplifiedRLStarter', 'BasicFCModel']


def _get_weight_features(model_repr, axis=None):
    weight_features = []
    for param in model_repr.values():
        pshape = len(param.shape)
        axis = tuple(range(-1, -1*(pshape), -1))
        weight_features += np.max(param, axis= axis).tolist()
        weight_features += np.mean(param, axis= axis).tolist()
        sub = np.mean(param, axis= axis) - np.median(param, axis= axis)
        weight_features += sub.tolist()
        weight_features += np.median(param, axis= axis).tolist()
        weight_features += np.sum(np.abs(param), axis= axis).tolist()
    return weight_features


def _get_eigen_features(model_repr):
    min_shape, params = 1, []
    for param in model_repr.values():
        if len(param.shape) > min_shape:
            reshaped_param = param.reshape(param.shape[0], -1)
            _, singular_values, _ = np.linalg.svd(reshaped_param, False)
            ssv = np.square(singular_values).flatten()
            params.append(ssv.max().item())
            params.append(ssv.mean().item())
            params.append((ssv.mean() - np.median(ssv)).item())
            params.append(np.median(ssv).item())
            params.append(ssv.sum().item())
            # params.extend(ssv.flatten().tolist())
    return params


def get_model_features(model, model_class, model_repr, infer=True):    
    features = []

    features.extend(_get_weight_features(model_repr))
    features.extend(_get_eigen_features(model_repr))

    if infer:
        return np.asarray([features])
    else:
        return features
