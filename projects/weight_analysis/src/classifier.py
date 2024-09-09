import numpy as np
from jsonargparse import ArgumentParser, ActionConfigFile
import logging
import feature_extractor as fe
import torch
import json
import jsonschema
import os
from sklearn.ensemble import GradientBoostingClassifier


def weight_analysis_detector(model_filepath,
                            result_filepath,
                            round_training_dataset_dirpath,
                            parameters_dirpath,
                            temp_var):

    logging.info('model_filepath = {}'.format(model_filepath))
    logging.info('result_filepath = {}'.format(result_filepath))
    logging.info('Using round_training_dataset_dirpath = {}'.format(round_training_dataset_dirpath))
    logging.info('Using parameters_dirpath = {}'.format(parameters_dirpath))
    logging.info('Setting temperature variable (parameter1) = {}'.format(temp_var))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info("Using compute device: {}".format(device))

    # extract class and features from predict model
    predict_model_class = fe.get_predict_model_class(model_filepath)
    predict_model_features = np.asarray([fe.get_model_features(model_filepath, predict_model_class)])

    # extract features for models in round10-training-dataset, should be shape (72, 663) for model A (Faster RCNN) (72, 487) for model B (SSD)
    X_filepath = os.path.join(parameters_dirpath, f'train_X_{predict_model_class}.npy')
    y_filepath = os.path.join(parameters_dirpath, f'train_y_{predict_model_class}.npy')
    if os.path.exists(X_filepath) and os.path.exists(y_filepath):
        X = np.load(X_filepath)
        y = np.load(y_filepath)
    else:
        logging.info('Need to extract features from training dataset')
        trainig_models_class_and_features = fe.get_features_and_labels_by_model_class(round_training_dataset_dirpath, predict_model_class)
        X = trainig_models_class_and_features['X']
        y = trainig_models_class_and_features['y']

    # load learned parameters from parameter_dirpath
    extra_X_filepath = os.path.join(parameters_dirpath, f'X_{predict_model_class}.npy')
    extra_y_filepath = os.path.join(parameters_dirpath, f'y_{predict_model_class}.npy')
    if os.path.exists(extra_X_filepath) and os.path.exists(extra_y_filepath):
        extra_X = np.load(extra_X_filepath)
        extra_y = np.load(extra_y_filepath)
        if X.shape[1] == extra_X.shape[1] and extra_X.shape[0] == extra_y.shape[0]:  #double check features and labels are extracted properly
            X = np.concatenate((X, extra_X), axis=0)
            y = np.concatenate((y, extra_y), axis=0)
    else:
        logging.info('No new learned parameters added, using only training dataset')

    # fit the features to classifiers
    classifiers = {'clf_A': GradientBoostingClassifier(n_estimators=1100, learning_rate=0.00225, max_depth=8, min_samples_split=17, subsample=.66, min_samples_leaf=4, max_features=230), 
                    'clf_B': GradientBoostingClassifier(n_estimators=900, learning_rate=0.003, max_depth=4, min_samples_split=11, subsample=.63, max_features=240)}

    clf = classifiers[f'clf_{predict_model_class}']
    clf.fit(X, y)

    if predict_model_class == 'A':
        trojan_probability = clf.predict_proba(predict_model_features)
    elif predict_model_class == 'B':
        log_probs = clf.predict_log_proba(predict_model_features)
        T = temp_var
        trojan_probability = np.exp(log_probs/T)/np.exp(log_probs/T).sum(axis=1).reshape([-1, 1])
    else:
        logging.warning('Not able to detect such model class')
        with open(result_filepath, 'w') as fh:
            fh.write("{}".format(0.50))
        return

    logging.info('Trojan Probability of this class {} model is: {}'.format(predict_model_class, trojan_probability[0, -1]))
    
    with open(result_filepath, 'w') as fh:
        fh.write("{}".format(trojan_probability[0, -1]))


def configure(output_parameters_dirpath,
              configure_models_dirpath):

    logging.info('Configuring detector parameters with models from ' + configure_models_dirpath)
    try:
        A = fe.get_features_and_labels_by_model_class(configure_models_dirpath, 'A')
        B = fe.get_features_and_labels_by_model_class(configure_models_dirpath, 'B')
    except:
        logging.info('There is problem extracting features from configure_models_dirpath')
        logging.info('Exit configuration mode')
        return
    
    os.makedirs(output_parameters_dirpath, exist_ok=True)
    logging.info('Writing configured parameter data to ' + output_parameters_dirpath)

    if A['X'].shape[0] != 0 and A['X'].shape[0] == A['y'].shape[0] and A['X'].shape[1] == 663:
        np.save(os.path.join(output_parameters_dirpath, 'X_A.npy'), A['X'])
        np.save(os.path.join(output_parameters_dirpath, 'y_A.npy'), A['y'])
    else:
        logging.info('There is problem in saving parameters for model A (fasterrcnn)')
    if B['X'].shape[0] != 0 and B['X'].shape[0] == B['y'].shape[0] and B['X'].shape[1] == 487:
        np.save(os.path.join(output_parameters_dirpath, 'X_B.npy'), B['X'])
        np.save(os.path.join(output_parameters_dirpath, 'y_B.npy'), B['y'])
    else:
        logging.info('There is problem in saving parameters for model B (ssd)')


if __name__ == '__main__':
    parser = ArgumentParser(description='Weight Analysis Classifier')
    parser.add_argument('--model_filepath', type=str, help='File path to the pytorch model file to be evaluated.')
    parser.add_argument('--result_filepath', type=str, help='File path to the file where output result should be written. After execution this file should contain a single line with a single floating point trojan probability.')
    parser.add_argument('--scratch_dirpath', type=str, help='File path to the folder where scratch disk space exists. This folder will be empty at execution start and will be deleted at completion of execution.')
    parser.add_argument('--examples_dirpath', type=str, help='File path to the directory containing json file(s) that contains the examples which might be useful for determining whether a model is poisoned.')

    parser.add_argument('--source_dataset_dirpath', type=str, help='File path to a directory containing the original clean dataset into which triggers were injected during training.', default=None)
    parser.add_argument('--round_training_dataset_dirpath', type=str, help='File path to the directory containing id-xxxxxxxx models of the current rounds training dataset.', default=None)

    parser.add_argument('--metaparameters_filepath', help='Path to JSON file containing values of tunable paramaters to be used when evaluating models.', action=ActionConfigFile)
    parser.add_argument('--schema_filepath', type=str, help='Path to a schema file in JSON Schema format against which to validate the config file.', default=None)
    parser.add_argument('--learned_parameters_dirpath', type=str, help='Path to a directory containing parameter data (model weights, etc.) to be used when evaluating models.  If --configure_mode is set, these will instead be overwritten with the newly-configured parameters.')

    parser.add_argument('--configure_mode', help='Instead of detecting Trojans, set values of tunable parameters and write them to a given location.', default=False, action="store_true")
    parser.add_argument('--configure_models_dirpath', type=str, help='Path to a directory containing models to use when in configure mode.')

    # these parameters need to be defined here, but their values will be loaded from the json file instead of the command line
    parser.add_argument('--parameter1', type=float, help='Tunable temperature variable for SSD model prediction only.')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s")
    logging.info("classifier.py launched")
    logging.info(args)


    # Validate config file against schema
    config_json = None
    if args.metaparameters_filepath is not None:
        if args.schema_filepath is not None:
            with open(args.metaparameters_filepath[0]()) as config_file:
                config_json = json.load(config_file)

            with open(args.schema_filepath) as schema_file:
                schema_json = json.load(schema_file)

            # this throws a fairly descriptive error if validation fails
            jsonschema.validate(instance=config_json, schema=schema_json)

    default_temp = 0.9
    if config_json is not None and config_json["parameter1"] is not None:
        default_temp = config_json["parameter1"]
        logging.info('Setting temperature variable (parameter1) from metaparemeter.json')
    elif args.parameter1 is not None:
        default_temp = args.parameter1
        
    if default_temp < 0.1 or default_temp > 1.5:
        logging.info('Tunable temperature variable (parameter1) given out of scope, resetting to 0.9')
        default_temp = 0.9

    if not args.configure_mode:
        if (args.model_filepath is not None and
            args.result_filepath is not None and
            args.round_training_dataset_dirpath is not None and
            args.learned_parameters_dirpath is not None):

            logging.info("Calling the trojan detector")
            logging.info('Calling the weight analysis classifier')

            weight_analysis_detector(args.model_filepath,
                                    args.result_filepath,
                                    args.round_training_dataset_dirpath,
                                    args.learned_parameters_dirpath,
                                    default_temp)
        else:
            logging.info("Required Evaluation-Mode parameters missing!")
    else:
        if (args.learned_parameters_dirpath is not None and
            args.configure_models_dirpath is not None):

            logging.info("Calling configuration mode")
            configure(args.learned_parameters_dirpath,
                      args.configure_models_dirpath)
        else:
            logging.info("Required Configure-Mode parameters missing!")