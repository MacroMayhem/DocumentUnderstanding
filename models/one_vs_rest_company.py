import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from joblib import dump, load
 #NOTE: Make sure that the class is labeled 'target' in the data file
'''tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=0)

# Average CV score on the training set was:0.6678710051073056
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=GradientBoostingClassifier(learning_rate=0.001, max_depth=10, max_features=0.5, min_samples_leaf=11, min_samples_split=9, n_estimators=100, subsample=0.15000000000000002)),
    FastICA(tol=0.4),
    StackingEstimator(estimator=GaussianNB()),
    KNeighborsClassifier(n_neighbors=86, p=1, weights="distance")
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
'''

class CompanyClassifier:
    def __init__(self):
        self.exported_pipeline = make_pipeline(
            StackingEstimator(estimator=GradientBoostingClassifier(learning_rate=0.001, max_depth=10, max_features=0.5, min_samples_leaf=11, min_samples_split=9, n_estimators=100, subsample=0.15000000000000002)),
            FastICA(tol=0.4),
            StackingEstimator(estimator=GaussianNB()),
            KNeighborsClassifier(n_neighbors=86, p=1, weights="distance"))

    def fit(self,training_features, training_target):
        self.exported_pipeline.fit(training_features, training_target)
        dump(self.exported_pipeline, './models/company.joblib')

    def load_model(self):
        try:
            self.exported_pipeline =  load('./models/company.joblib')
        except:
            raise Exception('Model needs to be trained and saved first')

    def predict(self,testing_features):
        return self.exported_pipeline.predict(testing_features)

    def probability(self,testing_features):
        return self.exported_pipeline.predict_proba(testing_features)