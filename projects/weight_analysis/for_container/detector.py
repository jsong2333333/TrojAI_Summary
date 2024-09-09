import json
import logging
import os
import joblib
from os import listdir, makedirs
from os.path import join, exists, basename

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm

from utils.abstract import AbstractDetector
from utils.models import load_model, load_models_dirpath, load_ground_truth

import torch

import feature_extractor as fe


# There's a difference in the filepath when loading data from learned_parameter folder in the container, deleting '.' in './learned_parameters'
param_grid = {'gbm__learning_rate': np.arange(.005, .0251, .005), 
              'gbm__n_estimators': range(500, 1001, 100), 
              'gbm__max_depth': range(2, 5), 
              'gbm__max_features': range(50, 651, 100),
              'gbm__min_samples_split': range(20, 101, 10),
              'gbm__min_samples_leaf': range(10, 51, 5)}

            
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Detector(AbstractDetector):
    def __init__(self, metaparameter_filepath, learned_parameters_dirpath, scale_parameters_filepath):
        """Detector initialization function.
        Args:
            metaparameter_filepath: str - File path to the metaparameters file.
            learned_parameters_dirpath: str - Path to the learned parameters directory.
            scale_parameters_filepath: str - File path to the scale_parameters file.
        """
        metaparameters = json.load(open(metaparameter_filepath, "r"))

        self.scale_parameters_filepath = scale_parameters_filepath
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

        # logging.info('Configuration Mode Currently Not Available')

        # return

        # Create the learned parameter folder if needed
        if not exists(self.learned_parameters_dirpath):
            makedirs(self.learned_parameters_dirpath)

        # List all available model
        model_path_list = sorted([join(models_dirpath, model) for model in listdir(models_dirpath)])
        logging.info(f"Loading %d models", len(model_path_list))

        model_dict, model_repr_dict, model_ground_truth_dict = load_models_dirpath(model_path_list)

        logging.info("Extracting features from models")
        X_s, y_s, X_l, y_l = fe.get_features_and_labels(model_dict, model_repr_dict, model_ground_truth_dict)

        logging.info("Automatically training GBM model")
        pipe = Pipeline(steps=[('gbm', GradientBoostingClassifier())])
                
        _, counts = np.unique(y_l, return_counts=True)
        kfold = min(min(counts), 5)
        if kfold < 2 or len(counts) != 2:
            logging.info(f'Not enough data points are given for auto-tuning the model.')
            return

        gsearch = GridSearchCV(estimator=pipe, param_grid=param_grid, scoring='neg_log_loss', n_jobs=-1, cv=kfold)
        gsearch.fit(X_s, y_s)
                
        model = gsearch.best_estimator_
        metaparams = gsearch.best_params_

        logging.info("Saving GBM model")
        joblib.dump(model, join(self.learned_parameters_dirpath, 'clf.joblib'))

        for k, v in metaparams.items():
            self.gbm_kwargs[k[5:]] = v

        self.write_metaparameters()
        logging.info("Configuration done!")

    def manual_configure(self, models_dirpath: str):
        """Configuration of the detector using the parameters from the metaparameters
        JSON file.
        Args:
            models_dirpath: str - Path to the list of model to use for training
        """

        # logging.info('Configuration Mode Currently Not Available')

        # return

        # Create the learned parameter folder if needed
        if not exists(self.learned_parameters_dirpath):
            makedirs(self.learned_parameters_dirpath)

        # List all available model
        model_path_list = sorted([join(models_dirpath, model) for model in listdir(models_dirpath)])
        logging.info(f"Loading %d models", len(model_path_list))

        model_dict, model_repr_dict, model_ground_truth_dict = load_models_dirpath(model_path_list)

        logging.info("Extracting features from models")
        X_s, y_s, X_l, y_l = fe.get_features_and_labels(model_dict, model_repr_dict, model_ground_truth_dict)

        logging.info("Fitting GBM model in manual mode")
        model = GradientBoostingClassifier(**self.gbm_kwargs, random_state=0)
        model.fit(X_s, y_s)

        logging.info("Saving GBM model")
        joblib.dump(model, join(self.learned_parameters_dirpath, 'clf.joblib'))

        self.write_metaparameters()
        logging.info("Configuration done!")

    def infer(
        self,
        model_filepath,
        result_filepath,
        scratch_dirpath,
        examples_dirpath,
        round_training_dataset_dirpath,
    ):
        """Method to predict wether a model is poisoned (1) or clean (0).
        Args:
            model_filepath:
            result_filepath:
            scratch_dirpath:
            examples_dirpath:
            round_training_dataset_dirpath:
        """

        logging.info("Loading model for prediction")
        model, model_repr, model_class = load_model(model_filepath)
        net = 'small_net' if int(model_class[3]) <= fe.NET_LEVEL else 'large_net'

        logging.info("Extracting model features")
        X = fe.get_model_features(model, model_repr, model_class)

        logging.info('Loading classifier')
        potential_reconfig_model_filepath = join(self.learned_parameters_dirpath, 'clf.joblib')
        if os.path.exists(potential_reconfig_model_filepath):
            clf = joblib.load(potential_reconfig_model_filepath)
        else:
            logging.info('Using original classifier')
            clf = joblib.load(join(fe.ORIGINAL_LEARNED_PARAM_DIR, f'{net}_detector.joblib'))
            # clf = torch.load(join(fe.ORIGINAL_LEARNED_PARAM_DIR, 'best_rnn_model.pt')).to(DEVICE)
    
        logging.info('Detecting trojan probability')
        try:
            trojan_probability = clf.predict_proba(X)[0, -1]
            # clf.eval()
            # X = torch.Tensor(X).float().permute(2, 0, 1).to(DEVICE)
            logging.info(f'X shape - {X.shape}')
            # with torch.no_grad():
                # output = clf(X)
                # output = torch.sigmoid(output.reshape(-1))
            # trojan_probability = output.item()
        except:
            logging.warning('Not able to detect such model class')
            with open(result_filepath, 'w') as fh:
                fh.write("{}".format(0.50))
            return

        logging.info('Trojan Probability of this model is: {}'.format(trojan_probability))
    
        with open(result_filepath, 'w') as fh:
            fh.write("{}".format(trojan_probability))
