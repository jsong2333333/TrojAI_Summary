import torch
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from tqdm import tqdm

DATA_PATH = '/scratch/data/TrojAI/round10-train-dataset/'

def get_all_features():
    all_features = []
    num_models = 143
    for model_num in range(num_models):
        model_filepath = _get_model_filepath(model_num)
        all_features.append(get_model_features(model_filepath)['features'])

    X = np.asarray(all_features)
    return X

def get_all_labels():
    metadata_filepath = os.path.join(DATA_PATH, "METADATA.csv")
    metadata = pd.read_csv(metadata_filepath)
    y = metadata['poisoned'].to_numpy()
    return y


def get_features_and_labels_by_model_class():
    model_A_features, model_B_features, model_A_labels, model_B_labels = [], [], [], []
    num_models = 144
    metadata_filepath = os.path.join(DATA_PATH, "METADATA.csv")
    metadata = pd.read_csv(metadata_filepath)
    labels = metadata['poisoned'].to_numpy()
    for model_num in tqdm(range(num_models)):
        model_filepath = _get_model_filepath(model_num)
        class_and_features_by_model = get_model_features(model_filepath)
        model_class, model_features = class_and_features_by_model['model_class'], class_and_features_by_model['features']
        if model_class == 'A':
            model_A_features.append(model_features)
            model_A_labels.append(labels[model_num])
        elif model_class == 'B':
            model_B_features.append(model_features)
            model_B_labels.append(labels[model_num])
    return {'model_A_features': np.asarray(model_A_features), 'model_B_features': np.asarray(model_B_features), 
            'model_A_labels': np.asarray(model_A_labels), 'model_B_labels': np.asarray(model_B_labels)}


def get_model_features(model_filepath):
    model = torch.load(model_filepath)
    model_backbone = model.backbone

    num_of_params = sum(p.numel() for p in model.parameters())/1000.0

    all_backbone_params = []
    for param in model_backbone.parameters():
        all_backbone_params.append(param.data.cpu().numpy())

    model_class, features = '', []
    if num_of_params == 41755.2860:  # model A
        model_class = 'A'
        features = _get_eigen_vals(all_backbone_params, 2, 3)
    elif num_of_params == 35641.8260:  # model B
        model_class = 'B'
        features = _get_eigen_vals(all_backbone_params, 2, 5)

    return {'model_class': model_class, 'features': features}


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


def _get_model_filepath(model_num: int) -> os.path:
    num_as_str = str(100000000 + model_num)[1:]
    model_id = f"id-{num_as_str}"
    return os.path.join(DATA_PATH, model_id, 'model.pt')


def bootstrap_performance(X, y, n=10, test_size=.2, eps=.01, model_class='A'):
    all_cross_entropy, all_accuracy = [], []
    for i in range(n):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=i)
        
        if model_class == 'A':
            clf = GradientBoostingClassifier(n_estimators=4000, learning_rate=0.0005, max_depth=3, subsample=0.7, max_features='sqrt', random_state=i, loss='log_loss')
        elif model_class == 'B':
            clf = GradientBoostingClassifier(n_estimators=4000, learning_rate=0.00007, max_depth=3, subsample=0.7, max_features='sqrt', random_state=i, loss='log_loss')
        
        clf.fit(X_train, y_train)
        
        all_cross_entropy.append(log_loss(y_test, clf.predict_proba(X_test), eps=eps))
        all_accuracy.append(clf.score(X_test, y_test))
    return all_cross_entropy, all_accuracy


if __name__ == "__main__":
    all_features_and_labels = get_features_and_labels_by_model_class()
    model_A_features, model_B_features, model_A_labels, model_B_labels = all_features_and_labels['model_A_features'], all_features_and_labels['model_B_features'], all_features_and_labels['model_A_labels'], all_features_and_labels['model_B_labels']
    print(model_A_features.shape, model_B_features.shape)

    # clf_A = GradientBoostingClassifier(n_estimators=4000, learning_rate=0.0005, max_depth=3, subsample=0.7, max_features='sqrt', random_state=0, loss='log_loss')
    clf_B = GradientBoostingClassifier(n_estimators=4000, learning_rate=0.00007, max_depth=3, subsample=0.7, max_features='sqrt', random_state=0, loss='log_loss')

    # cen_A, acc_A = bootstrap_performance(model_A_features, model_A_labels)
    cen_B, acc_B = bootstrap_performance(model_B_features, model_B_labels)

    print(np.mean([cen_B, acc_B], axis=-1))

    # clf_A.fit(model_A_features, model_A_labels)
    # clf_B.fit(model_B_features, model_B_labels)
    # dump(clf_A, 'classifier_model_A.joblib')
    # dump(clf_B, 'classifier_model_B.joblib')