import logging
import os
import json
import jsonpickle
import pickle
import numpy as np
import datasets
import torch
import transformers
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from utils.abstract import AbstractDetector
from utils.models import load_model, load_models_dirpath
import utils.qa_utils

import torch
import joblib
import feature_extractor as fe
from collections import OrderedDict


TUNE_PARAM_GRID = {'gbm__learning_rate': np.arange(.005, .0251, .005), 
                   'gbm__n_estimators': range(500, 1001, 100), 
                   'gbm__max_depth': range(2, 5), 
                   'gbm__max_features': range(50, 651, 100),
                   'gbm__min_samples_split': range(20, 101, 10),
                   'gbm__min_samples_leaf': range(10, 51, 5)}
TUNE_PARAM_NAMES = ['learning_rate', 'max_depth', 'n_estimators', 'max_features', 'min_samples_leaf', 'min_samples_split']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Detector(AbstractDetector):
    def __init__(self, metaparameter_filepath, learned_parameters_dirpath):
        """Detector initialization function.

        Args:
            metaparameter_filepath: str - File path to the metaparameters file.
            learned_parameters_dirpath: str - Path to the learned parameters directory.
        """
        metaparameters = json.load(open(metaparameter_filepath, "r"))

        self.metaparameter_filepath = metaparameter_filepath
        self.learned_parameters_dirpath = learned_parameters_dirpath

        self.gbm_kwargs = {}
        for k, v in metaparameters.items():
            if 'gbm_param' in k:
                # model_arch = k.split('_')[1]
                param_name = k.split('-')[-1]
                # self.gbm_kwargs[model_arch][param_name] = v
                for model_arch in fe.MODEL_ARCH:
                    if model_arch not in self.gbm_kwargs:
                        self.gbm_kwargs[model_arch] = {}
                    self.gbm_kwargs[model_arch][param_name] = v


    def write_metaparameters(self):
        metaparameters = {f'train_{model_arch}_gbm_param-{param_name}':value for model_arch, params in self.gbm_kwargs.items() for param_name, value in params.items()}

        with open(os.path.join(self.learned_parameters_dirpath, os.path.basename(self.metaparameter_filepath)), "w") as fp:
            fp.write(jsonpickle.encode(metaparameters, warn=True, indent=2))


    def automatic_configure(self, models_dirpath: str):
        """Configuration of the detector iterating on some of the parameters from the
        metaparameter file, performing a grid search type approach to optimize these
        parameters.

        Args:
            models_dirpath: str - Path to the list of model to use for training
        """
        # Create the learned parameter folder if needed
        if not os.path.exists(self.learned_parameters_dirpath):
            os.makedirs(self.learned_parameters_dirpath)

        # List all available model
        model_dirpath_list = sorted([os.path.join(models_dirpath, model) for model in os.listdir(models_dirpath)])
        logging.info(f"Loading %d models...", len(model_dirpath_list))
        
        X, y = {}, {}
        for model_dirpath in model_dirpath_list:
            model_filepath = os.path.join(model_dirpath, 'model.pt')
            model, model_repr, model_class = load_model(model_filepath)

            model_fe = fe.get_model_features(model, model_class, model_repr)
            
            gt_filepath = os.path.join(model_dirpath, 'ground_truth.csv')
            ground_truth = int(pd.read_csv(gt_filepath).columns[0])
            
            k = 'clf' #if model_class == 'SSD' else 'clfwa'
            X.setdefault(model_class, []).append(model_fe)
            y.setdefault(model_class, []).append(ground_truth)

        logging.info("Building GBM based on prvided parameters in AUTO CONFIGURATION mode.")
        
        for k, v in X.items():
            pipe = Pipeline(steps=[('gbm', GradientBoostingClassifier())])

            _, counts = np.unique(y[k], return_counts=True)

            kfold = min(min(counts), 5)
            if kfold < 2 or len(counts) != 2:
                logging.info(f'Not enough data points are given for auto-tuning the model for model.')
                return
            logging.info(f'cv kfold for this reconfig_clf is {kfold}')
                
            gsearch = GridSearchCV(estimator=pipe, param_grid=TUNE_PARAM_GRID, scoring='neg_log_loss', n_jobs=-1, cv=kfold)
            gsearch.fit(v, y[k])

            reconfig_clf = gsearch.best_estimator_
                
            reconfig_params = reconfig_clf.get_params()
            print(reconfig_params)

            for param_name in TUNE_PARAM_NAMES:
                self.gbm_kwargs[model_class][param_name] = reconfig_params[f'gbm__{param_name}']
            
            logging.info("Saving model...")
            joblib.dump(reconfig_clf, os.path.join(self.learned_parameters_dirpath, f'reconfig_{k}_clf.joblib'))
            
            logging.info("Writing new metaparameter file")
            self.write_metaparameters()
        
        logging.info("Auto Configuration finished.")


    def manual_configure(self, models_dirpath: str):
        """Configuration of the detector using the parameters from the metaparameters
        JSON file.

        Args:
            models_dirpath: str - Path to the list of model to use for training
        """
        # Create the learned parameter folder if needed
        if not os.path.exists(self.learned_parameters_dirpath):
            os.makedirs(self.learned_parameters_dirpath)

        # List all available model
        model_dirpath_list = sorted([os.path.join(models_dirpath, model) for model in os.listdir(models_dirpath)])
        logging.info(f"Loading %d models...", len(model_dirpath_list))
        
        X, y = {}, {}
        for model_dirpath in model_dirpath_list:
            model_filepath = os.path.join(model_dirpath, 'model.pt')
            model, model_repr, model_class = load_model(model_filepath)

            model_fe = fe.get_model_features(model, model_class, model_repr)
            
            gt_filepath = os.path.join(model_dirpath, 'ground_truth.csv')
            ground_truth = int(pd.read_csv(gt_filepath).columns[0])
            
            k = 'clf' #if model_class == 'SSD' else 'clfwa'
            X.setdefault(model_class, []).append(model_fe)
            y.setdefault(model_class, []).append(ground_truth)
        
            logging.info("Building GBM based on prvided parameters in MANUAL CONFIGURATION mode.")
        
            reconfig_clf = GradientBoostingClassifier().set_params(**self.gbm_kwargs[model_class]).fit(model_fe, ground_truth)
            
            logging.info(f"Saving model for model arch {model_class}...")
            joblib.dump(reconfig_clf, os.path.join(self.learned_parameters_dirpath, f'reconfig_{model_class}_clf.joblib'))
            
        logging.info("Manual Configuration finished.")


    def infer(
            self,
            model_filepath, 
            result_filepath, 
            scratch_dirpath, 
            examples_dirpath, 
            round_training_dataset_dirpath, 
            tokenizer_filepath
    ):
        """Method to predict whether a model is poisoned (1) or clean (0).

        Args:
            model_filepath:
            result_filepath:
            scratch_dirpath:
            examples_dirpath:
            round_training_dataset_dirpath:
        """

        logging.info("Using compute device: {}".format(device))

        logging.info("Loading model for prediction")
        # load the model
        model = torch.load(model_filepath)
        model_repr = OrderedDict({layer: tensor.to(device) for (layer, tensor) in model.state_dict().items() if layer.endswith('weight')})
        model_class = fe.LAYERS_TO_MODEL_ARCH[len(model_repr)]

        # logging.info('Loading feature selection layers')
        # arch_to_layers_pth = os.path.join(fe.ORIGINAL_LEARNED_PARAM_DIR, 'good_weight_keys.json')
        # chosen_layers = list(model_repr.keys())
        # if os.path.exists(arch_to_layers_pth):
        #     chosen_layers = json.load(open(arch_to_layers_pth, 'r'))[model_class]
        # model_repr = OrderedDict({layer: tensor.to(fe.DEVICE) for (layer, tensor) in model_repr.items() if layer in chosen_layers})
        # if model_class == fe.MODEL_ARCH[1]:
            # model_repr = OrderedDict({layer: tensor.to(fe.DEVICE) for (layer, tensor) in model.state_dict().items() if 'qa_outputs' in layer})
        # else:
            # model_repr = OrderedDict({layer: tensor.to(fe.DEVICE) for (layer, tensor) in model.state_dict().items() if 'position_ids' not in layer and 'embedding' in layer})

        logging.info("Getting model features")
        X = fe.get_model_features(model_repr)
        logging.info(f'X shape - {X.shape}')
        
        fe_imp = json.load(open(os.path.join(fe.ORIGINAL_LEARNED_PARAM_DIR, 'fe_imp.json'), 'r'))[model_class]

        logging.info('Loading classifier')
        potential_reconfig_model_filepath = os.path.join(self.learned_parameters_dirpath, f'reconfig_{model_class}_clf.joblib')
        if os.path.exists(potential_reconfig_model_filepath):
            clf = joblib.load(potential_reconfig_model_filepath)
        else:
            logging.info('Using original classifier')
            clf = joblib.load(os.path.join(fe.ORIGINAL_LEARNED_PARAM_DIR, f'{model_class}_fe_imp_dim0.joblib'))
    
        logging.info('Detecting trojan probability')
        try:
            X = X[:, np.asarray(fe_imp) != 0]
            trojan_probability = clf.predict_proba(X)[0, -1]
        except:
            logging.warning('Not able to detect such model class')
            with open(result_filepath, 'w') as fh:
                fh.write("{}".format(0.50))
            return

        logging.info('Trojan Probability of this model is: {}'.format(trojan_probability))
    
        with open(result_filepath, 'w') as fh:
            fh.write("{}".format(trojan_probability))
