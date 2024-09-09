import numpy as np
import os
import torch


ORIGINAL_LEARNED_PARAM_DIR = './learned_parameters'
MODEL_FILEDIR = '/scratch/data/TrojAI/cyber-network-c2-feb2024/models/' #'cyber-pdf-dec2022-train/models/'
METADATA_FILEPATH = '/scratch/data/TrojAI/cyber-network-c2-feb2024/METADATA.csv'
METADATADICT_FILEPATH = '/scratch/data/TrojAI/cyber-network-c2-feb2024/METADATA_DICTIONARY.csv'


def _get_keys(layer_num):
    return [f'layer{layer_num}.0.conv1.weight',
            f'layer{layer_num}.0.conv2.weight',
            f'layer{layer_num}.1.conv1.weight',
            f'layer{layer_num}.1.conv2.weight']


def _get_stats_from_weight_features(weight: np.ndarray, axis= (0,), normalized=False) -> list:
    params = []
    
    try:
        norm = torch.linalg.norm(weight, ord=2)
    except:
        norm = torch.linalg.norm(weight.reshape(weight.shape[0], -1), ord=2)
    
    if not normalized:
        norm = 1

    weight /= norm
    p_max = torch.amax(weight, dim=axis).flatten()
    p_mean = torch.mean(weight, dim=axis).flatten()
    # p_median = torch.median(weight, dim=axis) 
    # p_sub = p_mean - p_median
    p_sum = torch.sum(weight, dim=axis).flatten()

    try:
        p_rank = [np.linalg.norm(weight.cpu().numpy(), ord='fro')**2/np.linalg.norm(weight.cpu().numpy(), ord=2)**2]
        for ord in [2, 'fro', np.Inf, -np.Inf, 'nuc']:
            p_rank.append(np.linalg.norm(weight.cpu().numpy(), ord=ord))
    except:
        reshaped_weight = weight.reshape(weight.shape[0], -1).cpu().numpy()
        p_rank = [np.linalg.norm(reshaped_weight, ord='fro')**2/np.linalg.norm(reshaped_weight, ord=2)**2]
        for ord in [2, 'fro', np.Inf, -np.Inf, 'nuc']:
            p_rank.append(np.linalg.norm(reshaped_weight, ord=ord))
    
    for p in [p_max, p_mean, p_sum]:
        if isinstance(p, int):
            params.append(p)
        else:
            params.extend(p.cpu().tolist())
    params.extend(p_rank)
    return params


def get_model_features(model, infer=True, normalize=False, layer_num=1):
    params = []
    resnet18_keys = _get_keys(layer_num)
    model_arch = 18
    for layer, tensor in model.items():
        if layer in resnet18_keys:
            params.extend(_get_stats_from_weight_features(tensor, normalized=normalize, axis=list(range(len(tensor.shape)))[:1]))
        if 'layer4.2.conv2.weight' == layer:
            model_arch = 34
    if infer:
        return np.asarray([params]), model_arch
    else:
        return params, model_arch


def num_to_model_id(num):
    return 'id-' + str(100000000+num)[1:]


from sklearn.metrics import log_loss, roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

def bootstrap_performance(X, y, clf, n=10, test_size=.2):
    all_cross_entropy, all_accuracy, all_fe_imp = [], [], []
    for i in range(n):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=i)

        if np.unique(y_train).shape[0] == 1 or np.unique(y_test).shape[0] == 1:
            continue
        
        clf.set_params(random_state=i)            
        clf.fit(X_train, y_train)
        
        all_cross_entropy.append(log_loss(y_test, clf.predict_proba(X_test)))
        all_accuracy.append(clf.score(X_test, y_test))
        # all_accuracy.append(roc_auc_score(y_test, clf.predict_proba(X_test)[:, -1]))
        all_fe_imp.append(clf.feature_importances_)
    return all_cross_entropy, all_accuracy, all_fe_imp
    

if __name__ == '__main__':
    from tqdm import tqdm
    import pandas as pd

    poisoned = []
    for model_num in tqdm(range(1, 49, 1)):
        model_id = num_to_model_id(model_num)
        with open(os.path.join(MODEL_FILEDIR, model_id, "ground_truth.csv"), "r") as fp:
            model_ground_truth = fp.readlines()[0]
        poisoned.append(int(model_ground_truth))
    METADATA = pd.read_csv(METADATA_FILEPATH)
    METADATA['poisoned'] = poisoned
    # m = torch.load('/scratch/data/TrojAI/cyber-network-c2-feb2024/models/id-00000001/model.pt')

    # X, y = {}, {}
    # for layer_num in range(1, 5):
    #     resnet18_keys = [k for k in m.keys() if k.startswith(f'layer{layer_num}') and 'weight' in k and 'conv' in k]
    #     X[layer_num] = []
    #     y[layer_num] = []
    #     for model_num in tqdm(range(1, 49, 1)):
    #         model_id = num_to_model_id(model_num)
    #         model_filepath = os.path.join(MODEL_FILEDIR, model_id, 'model.pt')
    #         model = torch.load(model_filepath, map_location=torch.device('cuda:3'))
    #         # model_repr = OrderedDict({layer: tensor.numpy() for (layer, tensor) in model.items()})

    #         params = []
    #         for layer, tensor in model.items():
    #             if layer in resnet18_keys:
    #                 params.extend(_get_stats_from_weight_features(tensor, axis=list(range(len(tensor.shape)))[:1]))

    #         poisoned = METADATA[METADATA['model_name'] == model_id]['poisoned'].item()
    #         X[layer_num].append(params)
    #         y[layer_num].append(poisoned)
    #     np.save(f'X_layer{layer_num}.npy', X[layer_num])
    #     np.save(f'y_layer{layer_num}.npy', y[layer_num])

        # clf = GradientBoostingClassifier(learning_rate=.01, n_estimators=825, max_features=1000)#, max_depth= 5, min_samples_leaf= 16, min_samples_split= 80)  #0.07/0.035 - 550
        # subX, suby = X[layer_num], y[layer_num]
        # subX = np.asarray(subX)
        # # print(subX.shape)
        # cen, acc, fe_imps = bootstrap_performance(subX, suby, clf, n=10, test_size=.2) 
        # print(layer_num, np.mean(cen), np.mean(acc))
        
    import joblib
    print(joblib.__version__)
    clf = GradientBoostingClassifier(learning_rate=.01, n_estimators=825, max_features=1000)#, max_depth= 5, min_samples_leaf= 16, min_samples_split= 80)  #0.07/0.035 - 550
    for layer_num in range(1, 5):
        X, y = np.load(f'./X_layer{layer_num}.npy'), np.load(f'./y_layer{layer_num}.npy')
        joblib.dump(clf.fit(X, y), f'./learned_parameters/clf_layer{layer_num}.joblib')