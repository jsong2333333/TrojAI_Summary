import os
import json
import joblib

CLF_FILEDIR = '/scratch/jialin/object-detection-feb2023/weight_analysis/for_container/learned_parameters'
OUTPUT_FILEDIR = '/scratch/jialin/object-detection-feb2023/weight_analysis/extracted_source/'

param_names = ['learning_rate', 'max_depth', 'n_estimators', 'subsample', 'max_features', 'min_samples_leaf', 'min_samples_split']
schema_template = {
    "learning_rate": {
        "description": "How fast the classifier is learning.",
        "type": "number",
        "minimum": 0.001,
        "maximum": 1,
        "suggested_minimum": 0.005,
        "suggested_maximum": 0.05
      },
    "n_estimators": {
        "description": "The number of boosting stages.",
        "type": "integer",
        "minimum": 1,
        "maximum": 2000,
        "suggested_minimum": 200,
        "suggested_maximum": 1200
      },
    "max_depth": {
        "description": "The maximum depth of the individual regression estimators.",
        "type": "integer",
        "minimum": 1,
        "maximum": 10,
        "suggested_minimum": 1,
        "suggested_maximum": 5
      },
    "min_samples_split": {
        "description": "The minimum number of samples required to split an internal node.",
        "type": "integer",
        "minimum": 2,
        "maximum": 500,
        "suggested_minimum": 10,
        "suggested_maximum": 160
      },
    "min_samples_leaf": {
        "description": "The minimum number of samples required to be at a leaf node.",
        "type": "integer",
        "minimum": 1,
        "maximum": 500,
        "suggested_minimum": 2,
        "suggested_maximum": 80
      },
    "max_features": {
        "description": "The number of features to consider when looking for the best split.",
        "type": ["integer", "string"],
        "minimum": 1,
        "maximum": 900,
        "suggested_minimum": 20,
        "suggested_maximum": 700
      },
    "subsample": {
        "description": "The fraction of samples to be used for fitting the individual base learners.",
        "type": "number",
        "minimum": 0.01,
        "maximum": 1,
        "suggested_minimum": 0.05,
        "suggested_maximum": 0.98
      }}

METAPARAM, SCHEMA = {}, {}

for clf_name in os.listdir(CLF_FILEDIR):
    clf_path = os.path.join(CLF_FILEDIR, clf_name)
    clf = joblib.load(clf_path)

    model_arch = clf_name.split('_')[0]
    for k, v in clf.get_params().items():
        if k in param_names:
            key_in_schema = f"train_{model_arch}_gbm_param-{k}"
            METAPARAM[key_in_schema] = v
            SCHEMA[key_in_schema] = schema_template[k]

with open(os.path.join(OUTPUT_FILEDIR, 'metaparameters.json'), 'w') as outfile:
    json.dump(METAPARAM, outfile, indent=4, separators=(',', ': '), sort_keys=True)
with open(os.path.join(OUTPUT_FILEDIR, 'metaparameters_schema.json'), 'w') as outfile:
    json.dump(SCHEMA, outfile, indent=4, separators=(',', ': '), sort_keys=True)