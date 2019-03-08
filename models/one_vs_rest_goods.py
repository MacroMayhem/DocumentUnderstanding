import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline, make_union
from sklearn.svm import LinearSVC
from tpot.builtins import StackingEstimator
from joblib import dump, load

# NOTE: Make sure that the class is labeled 'target' in the data file
'''tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=0)

# Average CV score on the training set was:0.6724628204108155
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=KNeighborsClassifier(n_neighbors=1, p=1, weights="uniform")),
    StackingEstimator(estimator=LinearSVC(C=25.0, dual=False, loss="squared_hinge", penalty="l2", tol=0.1)),
    StackingEstimator(estimator=ExtraTreesClassifier(bootstrap=False, criterion="entropy", max_features=0.9500000000000001, min_samples_leaf=12, min_samples_split=11, n_estimators=100)),
    KNeighborsClassifier(n_neighbors=63, p=2, weights="distance")
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
'''
class GoodsClassifier:
    def __init__(self):
        self.exported_pipeline = make_pipeline(
            StackingEstimator(estimator=KNeighborsClassifier(n_neighbors=1, p=1, weights="uniform")),
            StackingEstimator(estimator=LinearSVC(C=25.0, dual=False, loss="squared_hinge", penalty="l2", tol=0.1)),
            StackingEstimator(estimator=ExtraTreesClassifier(bootstrap=False, criterion="entropy", max_features=0.9500000000000001, min_samples_leaf=12, min_samples_split=11, n_estimators=100)),
            KNeighborsClassifier(n_neighbors=63, p=2, weights="distance"))

    def fit(self, training_features, training_target):
        self.exported_pipeline.fit(training_features, training_target)
        dump(self.exported_pipeline, './models/goods.joblib')

    def load_model(self):
        try:
            self.exported_pipeline = load('./models/goods.joblib')
        except:
            raise Exception('Model needs to be trained and saved first')
    def predict(self, testing_features):
        return self.exported_pipeline.predict(testing_features)

    def probability(self, testing_features):
        return self.exported_pipeline.predict_proba(testing_features)
