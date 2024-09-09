import os
import numpy as np
import torch

PATH = '/scratch/jialin/cyber-pdf-dec2022/projects/weight_analysis/for_container/learned_parameters'
DATA_PATH = '/scratch/data/TrojAI/cyber-pdf-dec2022-train/models/'
NET_LEVEL = 7
ORIGINAL_LEARNED_PARAM_DIR = './learned_parameters'


def get_features_and_labels(model_dict: dict, model_repr_dict: dict, model_ground_truth_dict: dict):
    X, y = {'small_net':[], 'large_net':[]}, {'small_net':[], 'large_net':[]}

    for model_arch, model_reprs in model_repr_dict.items():
        model_ground_truth = model_ground_truth_dict[model_arch]
        models = model_dict[model_arch]

        net = 'small_net'
        if int(model_arch[3]) > NET_LEVEL:
            net = 'large_net'

        y[net] += model_ground_truth

        for model, model_repr in zip(models, model_reprs):
            X[net].append(get_model_features(model, model_repr, model_arch, infer=False))
    
    return np.asarray(X['small_net']), np.asarray(y['small_net']), np.asarray(X['large_net']), np.asarray(y['large_net'])


def get_model_features(model, model_repr: dict, model_class: str, infer=True):
    features = []
    ok = _get_ordered_key(model_repr)

    if int(model_class[3]) > NET_LEVEL:  #larger nets (excluding Net2)
        norm_mul_weight = _get_multiplied_weight_features(model_repr, ok, normalized=True)
        features += norm_mul_weight.flatten().tolist()
        features += _get_fft_from_weight_features(norm_mul_weight)
        mul_weight = _get_multiplied_weight_features(model_repr, ok[1:])
        features += _get_eigen_from_weight_features(mul_weight)
        features += _get_stats_from_weight_features(mul_weight)
    else:
        final_layer = ok[0].split('.')[0]
        for k in ['fc1.weight', f'{final_layer}.weight']:
            features += _get_stats_from_weight_features(model_repr[k], normalized=True)
        # mul_weight = _get_multiplied_weight_features(model_repr, ok, normalized=False)
        # features += mul_weight.flatten().tolist()
        norm_mul_weight = _get_multiplied_weight_features(model_repr, ok, normalized=True)
        features += norm_mul_weight.flatten().tolist()
        # features += _get_stats_from_weight_features(norm_mul_weight, axis=(0, 1))
        # features += _get_fft_from_weight_features(mul_weight)
        # no_final_layer_mul_weight = _get_multiplied_weight_features(model_repr, ok[1:], normalized=True)
        # features += _get_eigen_from_weight_features(no_final_layer_mul_weight, 0, 38)
        # features.append(mul_weight.flatten().tolist()[200:260])
        # features.append(norm_mul_weight.flatten().tolist()[200:260])

    feature_ind = np.load(os.path.join(ORIGINAL_LEARNED_PARAM_DIR, 'ind.npy'))
    features = np.asarray(features)[feature_ind].tolist()

    if infer:
        return np.asarray([features])
    else:
        return features


def _load_feature_ind(param_dirpath: str) -> list:
    return np.load(os.path.join(param_dirpath, 'param_ind.npy')).tolist()


def _get_ordered_key(model_repr: dict, weight_or_bias='weight', reversed=True) -> list:
    keys = [k for k in model_repr.keys() if weight_or_bias in k]
    if reversed:
        return keys[::-1]
    else:
        return keys


def _get_multiplied_weight_features(model_repr: dict, ordered_keys: list, normalized=False, ord=2) -> np.ndarray:
    weight = None
    for ok in ordered_keys:
        if normalized:
            weight = model_repr[ok] / np.linalg.norm(model_repr[ok], ord=ord) if weight is None else (weight @ (model_repr[ok]/ np.linalg.norm(model_repr[ok], ord=ord)))  
        else:
            weight = model_repr[ok] if weight is None else weight @ model_repr[ok]
    return weight


def _get_stats_from_weight_features(weight: np.ndarray, axis= (0,), normalized=False) -> list:
    params = []
    
    try:
        norm = np.linalg.norm(weight, ord=2)
    except:
        norm = np.linalg.norm(weight.reshape(weight.shape[0], -1), ord=2)
    
    if not normalized:
        norm = 1

    weight /= norm
    p_max = np.amax(weight, axis=axis) 
    p_mean = np.mean(weight, axis=axis)
    p_median = np.median(weight, axis=axis) 
    p_sub = p_mean - p_median
    p_sum = np.sum(weight, axis=axis)

    try:
        p_rank = [np.linalg.norm(weight, ord='fro')**2/np.linalg.norm(weight, ord=2)**2]
        for ord in [2, 'fro', np.Inf, -np.Inf, 'nuc']:
            p_rank.append(np.linalg.norm(weight, ord=ord))
    except:
        reshaped_weight = weight.reshape(weight.shape[0], -1)
        p_rank = [np.linalg.norm(reshaped_weight, ord='fro')**2/np.linalg.norm(reshaped_weight, ord=2)**2]
        for ord in [2, 'fro', np.Inf, -np.Inf, 'nuc']:
            p_rank.append(np.linalg.norm(reshaped_weight, ord=ord))
    
    if len(weight.shape) - len(axis) == 0:
        params = [p_max.tolist(), p_mean.tolist(), p_sub.tolist(), p_median.tolist(), p_sum.tolist()] + p_rank
    else:
        params = p_max.tolist() + p_mean.tolist() + p_sub.tolist() + p_median.tolist() + p_sum.tolist() + p_rank
    return params


def _get_fft_from_weight_features(weight: np.ndarray) -> list:
    ft = np.fft.ifftshift(weight)
    ft = np.fft.fft2(ft)
    ft = np.fft.fftshift(ft)
    return np.abs(ft.flatten()).tolist()


def _get_eigen_from_weight_features(weight: np.ndarray, sind=0, eind=45) -> list:
    _, s, _ = np.linalg.svd(weight)
    return s[sind:eind].tolist()


def _get_sample_output(model, input_data: torch.FloatTensor) -> list:
    output = model(input_data)
    return output[:, 0].tolist()


if __name__ =='__main__':
    from utils.models import load_model, load_models_dirpath, load_ground_truth

    models_dirpath = sorted([os.path.join(DATA_PATH, model_dirpath) for model_dirpath in os.listdir(DATA_PATH)])
    model_dict, model_repr_dict, model_ground_truth_dict = load_models_dirpath(models_dirpath)
    X_s, y_s, X_l, y_l = get_features_and_labels(model_dict, model_repr_dict, model_ground_truth_dict)

    # model_arch = []
    # for k, v in model_dict.items():
    #     model_arch += [k]*len(v)

    print(X_s.shape)
    print(X_s[0, :100])

    # OUTPUT_DIR = '/scratch/jialin/cyber-pdf-dec2022/projects/weight_analysis/extracted_source'
    # np.save(os.path.join(OUTPUT_DIR, 'X.npy'), X_s)
    # np.save(os.path.join(OUTPUT_DIR, 'y.npy'), y_s)
    # np.save(os.path.join(OUTPUT_DIR, 'fe_arch.npy'), model_arch)
    # np.save(os.path.join(OUTPUT_DIR, 'fe_X_l.npy'), X_l)
    # np.save(os.path.join(OUTPUT_DIR, 'fe_y_l.npy'), y_l)