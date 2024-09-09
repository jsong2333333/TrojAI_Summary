import numpy as np
import os


ORIGINAL_LEARNED_PARAM_DIR = './learned_parameters'
FE_IMP_MEAN = np.load(os.path.join(ORIGINAL_LEARNED_PARAM_DIR, 'fe_imps_mean.npy'))

def get_model_features(model_repr: dict, infer=True, normalize=False, layers='all', fe_imp=None):
    # reversed_order_key = [k for k in model_repr.keys() if 'weight' in k][::-1]
    # final_layer = reversed_order_key[0].split('.')[0]

    # features = []
    # if layers == 'all':
    #     weight_keys = [f'fc{n}.weight' for n in range(1, 5)]
    # else:
    #     weight_keys = ['fc1.weight', f'{final_layer}.weight']
    # for k in weight_keys:
    #     mat = model_repr[k]
    #     norm = 1
    #     if normalize:
    #         norm = np.linalg.norm(mat)
    #     _, s, _ = np.linalg.svd(mat.reshape(mat.shape[0], -1)/norm)
    #     features.extend(s.flatten().tolist()[:100])
    
    num_layer = len(model_repr.items())//2
    for nl in range(num_layer, 0, -1):
        weight_layer_name = f'fc{nl}.weight'
        if nl == num_layer:
            mat = model_repr[weight_layer_name]
        else:
            mat = mat @ model_repr[weight_layer_name]
    norm = np.linalg.norm(mat, ord=2) if normalize else 1.
    mat = mat/norm
    features = mat.flatten()
    if fe_imp == 1128:
        features = features[FE_IMP_MEAN>=1e-4]
    elif fe_imp == 799:
        features = features[FE_IMP_MEAN>=2e-4]

    if infer:
        return np.asarray([features])
    else:
        return features