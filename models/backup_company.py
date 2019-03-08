import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator

# NOTE: Make sure that the class is labeled 'target' in the data file
#tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
#features = tpot_data.drop('target', axis=1).values
#training_features, testing_features, training_target, testing_target = \
#            train_test_split(features, tpot_data['target'].values, random_state=0)

# Average CV score on the training set was:0.6137586794796581

class CompanyClassifier:
    def __init__(self):
        self.exported_pipeline = make_pipeline(
            StackingEstimator(estimator=GradientBoostingClassifier(learning_rate=0.001, max_depth=3, max_features=0.5, min_samples_leaf=11, min_samples_split=8, n_estimators=100, subsample=0.45)),
            KNeighborsClassifier(n_neighbors=50, p=2, weights="distance")
            )

        #exported_pipeline.fit(training_features, training_target)
        #results = exported_pipeline.predict(testing_features)

    def fit(self,training_features, training_target):
        self.exported_pipeline.fit(training_features, training_target)

    def predict(self,testing_features):
        return self.exported_pipeline.predict(testing_features)

    def probability(self,testing_features):
        return self.exported_pipeline.predict_proba(testing_features)
