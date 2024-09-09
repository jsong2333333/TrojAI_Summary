from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
import os
import numpy as np
import pandas as pd

OUTPUT_FILEDIR = '/scratch/jialin/cyber-pdf-dec2022/projects/weight_analysis/extracted_source'
X, y = np.load(os.path.join(OUTPUT_FILEDIR, 'X.npy')), np.load(os.path.join(OUTPUT_FILEDIR, 'y.npy'))

clf = GradientBoostingClassifier(learning_rate=.02, n_estimators=875)
param={'max_depth': range(3, 5), 'min_samples_leaf': range(2, 35, 2), 'min_samples_split': range(4, 85, 4), 'max_features': range(5, 66, 5)}
# param.update({'learning_rate':np.arange(.005, .0251, .001), 'n_estimators':range(200, 1201, 25)})
gsearch = GridSearchCV(estimator=clf, param_grid=param, scoring=['neg_log_loss', 'accuracy'], n_jobs=10, cv=5, refit=False);
gsearch.fit(X, y)
gsearch_result = pd.DataFrame(gsearch.cv_results_).sort_values(by=['rank_test_neg_log_loss','rank_test_accuracy'])
gsearch_result.to_csv(os.path.join(OUTPUT_FILEDIR, 'gsearch_result_with_last_layer.csv'))