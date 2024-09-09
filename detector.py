# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.


import logging
import os
import json
import jsonpickle
import pickle
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from utils.abstract import AbstractDetector
from utils.models import load_model, load_models_dirpath

import torch
import torch_ac
import gym
from gym_minigrid.wrappers import ImgObsWrapper

import joblib
import feature_extractor as fe


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


    def inference_on_example_data(self, model, examples_dirpath):
        """Method to demonstrate how to inference on a round's example data.

        Args:
            model: the pytorch model
            examples_dirpath: the directory path for the round example data
        """

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info("Using compute device: {}".format(device))

        model.to(device)
        model.eval()

        preprocess = torch_ac.format.default_preprocess_obss

        # Utilize open source minigrid environment model was trained on
        env_string_filepath = os.path.join(examples_dirpath, 'env-string.txt')
        with open(env_string_filepath) as env_string_file:
            env_string = env_string_file.readline().strip()
        logging.info('Evaluating on {}'.format(env_string))

        # Number of episodes to run
        episodes = 100

        env_perf = {}

        # Run episodes through an environment to collect what may be relevant information to trojan detection
        # Construct environment and put it inside a observation wrapper
        env = ImgObsWrapper(gym.make(env_string))
        obs = env.reset()
        obs = preprocess([obs], device=device)

        final_rewards = []
        with torch.no_grad():
            # Episode loop
            for _ in range(episodes):
                done = False
                # Use env observation to get action distribution
                dist, value = model(obs)
                # Per episode loop
                while not done:
                    # Sample from distribution to determine which action to take
                    action = dist.sample()
                    action = action.cpu().detach().numpy()
                    # Use action to step environment and get new observation
                    obs, reward, done, info = env.step(action)
                    # Preprocessing function to prepare observation from env to be given to the model
                    obs = preprocess([obs], device=device)
                    # Use env observation to get action distribution
                    dist, value = model(obs)

                # Collect episode performance data (just the last reward of the episode)
                final_rewards.append(reward)
                # Reset environment after episode and get initial observation
                obs = env.reset()
                obs = preprocess([obs], device=device)

        # Save final rewards
        env_perf['final_rewards'] = final_rewards


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

        logging.info("Using compute device: {}".format(device))

        logging.info("Loading model for prediction")
        # load the model
        model, model_repr, model_class = load_model(model_filepath)

        logging.info("Getting model features")
        X = fe.get_model_features(model, model_class, model_repr)
        logging.info(f'X shape - {X.shape}')

        related_ind_filepath = os.path.join(fe.ORIGINAL_LEARNED_PARAM_DIR, 'fe_ind2.json')
        if os.path.exists(related_ind_filepath):
            extracted_inds = json.load(open(related_ind_filepath, 'r'))[model_class]
            X = X[:, extracted_inds]
            logging.info(f'reduced fe X shape - {X.shape}')
        # CHANGE CLASSIFIER!!!

        logging.info('Loading classifier')
        potential_reconfig_model_filepath = os.path.join(self.learned_parameters_dirpath, f'reconfig_{model_class}_clf.joblib')
        if os.path.exists(potential_reconfig_model_filepath):
            clf = joblib.load(potential_reconfig_model_filepath)
        else:
            logging.info('Using original classifier')
            clf = joblib.load(os.path.join(fe.ORIGINAL_LEARNED_PARAM_DIR, f'{model_class}_clf3.joblib'))
    
        logging.info('Detecting trojan probability')
        try:
            trojan_probability = clf.predict_proba(X)[0, -1]
        except:
            logging.warning('Not able to detect such model class')
            with open(result_filepath, 'w') as fh:
                fh.write("{}".format(0.50))
            return

        logging.info('Trojan Probability of this model is: {}'.format(trojan_probability))
    
        with open(result_filepath, 'w') as fh:
            fh.write("{}".format(trojan_probability))
