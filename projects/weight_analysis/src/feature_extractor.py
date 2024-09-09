import torch
import os
import numpy as np
import json

DATA_PATH = '/scratch/data/TrojAI/round10-train-dataset/'
NUM_MODELS = 144
model_arch_to_AB = {'fasterrcnn': 'A', 'ssd': 'B'}
model_class_to_layers = {'A':[2, 3], 'B':[2, 5]}

def get_features_and_labels_by_model_class(dp, class_A_or_B):
    ret_dir = {'X': [], 'y': []}
    child_dirs = os.listdir(dp)
    for child_dir in child_dirs:
        model_filedir = os.path.join(dp, child_dir)
        if os.path.isdir(model_filedir):
            model_filepath = os.path.join(model_filedir, 'model.pt')
            json_filepath = os.path.join(model_filedir, 'config.json')
            model_arch = _get_model_arch(json_filepath)
            model_class = model_arch_to_AB[model_arch]
            if model_class == class_A_or_B:
                model_features = get_model_features(model_filepath, model_class)
                model_label = _get_model_label(json_filepath)
                ret_dir['X'].append(model_features)
                ret_dir['y'].append(model_label)
    for k, v in ret_dir.items():
        ret_dir[k] = np.asarray(v)
    return ret_dir


def get_model_features(model_filepath, model_class):
    with torch.no_grad():
        model = torch.load(model_filepath)
    model_backbone = model.backbone

    all_backbone_params = []
    for param in model_backbone.parameters():
        all_backbone_params.append(param.data.cpu().numpy())
    
    layers = model_class_to_layers[model_class]
    features = _get_eigen_vals(all_backbone_params, layers[0], layers[1])
    weight_features = _get_weight_features(all_backbone_params)
    features += weight_features
    return features


def get_predict_model_class(model_filepath):
    with torch.no_grad():
        model = torch.load(model_filepath)
    num_of_params = sum(p.numel() for p in model.parameters())/1000.0
    if num_of_params == 41755.2860:  # model A rcnn
        model_class = 'A'
    elif num_of_params == 35641.8260:  # model B ssd
        model_class = 'B'
    return model_class


def _get_model_arch(json_filepath) -> bool:
    with open(json_filepath, 'r') as f:
        config = json.loads(f.read())
    return config['py/state']['model_architecture']


def _get_model_label(json_filepath) -> bool:
    with open(json_filepath, 'r') as f:
        config = json.loads(f.read())
    return config['py/state']['poisoned']


def _get_weight_features(all_backbone_params):
    weight_features = []
    for backbone_params in all_backbone_params:
        if len(backbone_params.shape)>2:
            weight_features += backbone_params.max(axis=(0, 1, 2)).tolist()
            weight_features += backbone_params.mean(axis=(0, 1, 2)).tolist()
            sub = backbone_params.mean(axis=(0, 1, 2)) - np.median(backbone_params, axis=(0, 1, 2))
            weight_features += sub.tolist()
            weight_features += np.median(backbone_params, axis=(0, 1, 2)).tolist()
            weight_features += backbone_params.sum(axis=(0, 1, 2)).tolist()
    return weight_features


def _get_eigen_vals(all_backbone_params, idx_low=0, idx_high=3):
    features = []
    num_layers = 0
    for backbone_params in all_backbone_params:
        if len(backbone_params.shape) > 2:
            if num_layers >= idx_low and num_layers <= idx_high:
                reshaped_params = backbone_params.reshape(backbone_params.shape[0], -1)
                _, singular_values, _ = np.linalg.svd(reshaped_params,False)
                squared_singular_values = singular_values**2
                features += squared_singular_values.tolist()
        num_layers += 1
    return features


if __name__ == "__main__":
    A_features_and_labels = get_features_and_labels_by_model_class('/scratch/data/TrojAI/round10-train-dataset/', 'A')
    B_features_and_labels = get_features_and_labels_by_model_class('/scratch/data/TrojAI/round10-train-dataset/', 'B')
    model_A_features, model_B_features, model_A_labels, model_B_labels = A_features_and_labels['X'], B_features_and_labels['X'], A_features_and_labels['y'], B_features_and_labels['y']
    print('')

    # np.save('/scratch/jialin/round-10/projects/weight_analysis/extracted_source/X_A.npy', model_A_features)
    # np.save('/scratch/jialin/round-10/projects/weight_analysis/extracted_source/X_B.npy', model_B_features)
    # np.save('/scratch/jialin/round-10/projects/weight_analysis/extracted_source/y_A.npy', model_A_labels)
    # np.save('/scratch/jialin/round-10/projects/weight_analysis/extracted_source/y_B.npy', model_B_labels)

    # clf_A = GradientBoostingClassifier(n_estimators=1100, learning_rate=0.00225, max_depth=8, min_samples_split=17, subsample=.66, min_samples_leaf=4, max_features=230)
    # clf_B = GradientBoostingClassifier(n_estimators=900, learning_rate=0.003, max_depth=4, min_samples_split=11, subsample=.63, max_features=240)
    # clf_A.fit(model_A_features, model_A_labels)
    # clf_B.fit(model_B_features, model_B_labels)
    # dump(clf_A, '/scratch/jialin/round-10/projects/weight_analysis/extracted_source/classifier_model_A.joblib')
    # dump(clf_B, '/scratch/jialin/round-10/projects/weight_analysis/extracted_source/classifier_model_B.joblib')