import json
import logging
import os
import pickle
from os import listdir, makedirs
from os.path import join, exists, basename

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm

from utils.abstract import AbstractDetector
from utils.flatten import flatten_model, flatten_models
from utils.healthchecks import check_models_consistency
from utils.models import create_layer_map, load_model, load_models_dirpath
from utils.padding import create_models_padding, pad_model
from utils.reduction import (
    fit_feature_reduction_algorithm,
    use_feature_reduction_algorithm,
)

import feature_extractor as fe
import joblib
import torch

class Detector(AbstractDetector):
    def __init__(self, metaparameter_filepath, learned_parameters_dirpath):
        """Detector initialization function.

        Args:
            metaparameter_filepath: str - File path to the metaparameters file.
            learned_parameters_dirpath: str - Path to the learned parameters directory.
            scale_parameters_filepath: str - File path to the scale_parameters file.
        """
        metaparameters = json.load(open(metaparameter_filepath, "r"))

        self.metaparameter_filepath = metaparameter_filepath
        self.learned_parameters_dirpath = learned_parameters_dirpath

        self.gbm_kwargs = {k[16:]: v for k, v in metaparameters.items() if k.startswith('train_gbm_param')}

    def write_metaparameters(self):
        metaparameters = {f'train_gbm_param_{k}': v for k, v in self.gbm_kwargs.items()}

        with open(join(self.learned_parameters_dirpath, basename(self.metaparameter_filepath)), "w") as fp:
            json.dump(metaparameters, fp)

    def automatic_configure(self, models_dirpath: str):
        """Configuration of the detector iterating on some of the parameters from the
        metaparameter file, performing a grid search type approach to optimize these
        parameters.

        Args:
            models_dirpath: str - Path to the list of model to use for training
        """
        # if not exists(self.learned_parameters_dirpath):
        #     makedirs(self.learned_parameters_dirpath)

        # # List all available model
        # model_path_list = sorted([join(models_dirpath, model) for model in listdir(models_dirpath)])
        # logging.info(f"Loading %d models", len(model_path_list))

        # model_dict, model_repr_dict, model_ground_truth_dict = load_models_dirpath(model_path_list)

        # logging.info("Extracting features from models")
        # X_s, y_s, X_l, y_l = fe.get_features_and_labels(model_dict, model_repr_dict, model_ground_truth_dict)

        # logging.info("Automatically training GBM model")
        # pipe = Pipeline(steps=[('gbm', GradientBoostingClassifier())])
                
        # _, counts = np.unique(y_l, return_counts=True)
        # kfold = min(min(counts), 5)
        # if kfold < 2 or len(counts) != 2:
        #     logging.info(f'Not enough data points are given for auto-tuning the model.')
        #     return

        # gsearch = GridSearchCV(estimator=pipe, param_grid=param_grid, scoring='neg_log_loss', n_jobs=-1, cv=kfold)
        # gsearch.fit(X_s, y_s)
                
        # model = gsearch.best_estimator_
        # metaparams = gsearch.best_params_

        # logging.info("Saving GBM model")
        # joblib.dump(model, join(self.learned_parameters_dirpath, 'clf.joblib'))

        # for k, v in metaparams.items():
        #     self.gbm_kwargs[k[5:]] = v

        # self.write_metaparameters()
        # logging.info("Configuration done!")

    def manual_configure(self, models_dirpath: str):
        """Configuration of the detector using the parameters from the metaparameters
        JSON file.

        Args:
            models_dirpath: str - Path to the list of model to use for training
        """
        # # Create the learned parameter folder if needed
        # if not exists(self.learned_parameters_dirpath):
        #     makedirs(self.learned_parameters_dirpath)

        # # List all available model
        # model_path_list = sorted([join(models_dirpath, model) for model in listdir(models_dirpath)])
        # logging.info(f"Loading %d models", len(model_path_list))

        # model_dict, model_repr_dict, model_ground_truth_dict = load_models_dirpath(model_path_list)

        # logging.info("Extracting features from models")
        # X_s, y_s, X_l, y_l = fe.get_features_and_labels(model_dict, model_repr_dict, model_ground_truth_dict)

        # logging.info("Fitting GBM model in manual mode")
        # model = GradientBoostingClassifier(**self.gbm_kwargs, random_state=0)
        # model.fit(X_s, y_s)

        # logging.info("Saving GBM model")
        # joblib.dump(model, join(self.learned_parameters_dirpath, 'clf.joblib'))

        # self.write_metaparameters()
        # logging.info("Configuration done!")
        

    def infer(
        self,
        model_filepath,
        result_filepath,
        scratch_dirpath,
        examples_dirpath,
        round_training_dataset_dirpath,
    ):
        """Method to predict whether a model is poisoned (1) or clean (0).

        Args:
            model_filepath:
            result_filepath:
            scratch_dirpath:
            examples_dirpath:
            round_training_dataset_dirpath:
        """

        logging.info("Loading model for prediction")
        model = torch.load(model_filepath, map_location=torch.device('cuda'))
        layer_num = 1

        logging.info("Extracting model features")
        X, model_arch = fe.get_model_features(model, normalize=False, layer_num=layer_num)
        # if os.path.exists(os.path.join(fe.ORIGINAL_LEARNED_PARAM_DIR, 'fe_imp.npy')):
        #     idx = np.load(os.path.join(fe.ORIGINAL_LEARNED_PARAM_DIR, 'fe_imp.npy'))
        #     X = X[:, idx]
        logging.info(f'X shape - {X.shape}')
        
        logging.info('Loading classifier')
        logging.info('Using original classifier')
        clf = joblib.load(join(fe.ORIGINAL_LEARNED_PARAM_DIR, f'clf_layer{layer_num}_{model_arch}.joblib'))
    
        logging.info('Detecting trojan probability')
        try:
            trojan_probability = clf.predict_proba(X)[0, 0]
        except:
            logging.warning('Not able to detect such model class')
            with open(result_filepath, 'w') as fh:
                fh.write("{}".format(0.50))
            return

        logging.info('Trojan Probability of this model is: {}'.format(trojan_probability))
    
        with open(result_filepath, 'w') as fh:
            fh.write("{}".format(trojan_probability))
