import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import pickle
from tpot import TPOTClassifier

import tpot

class Classifier:

    def tpot_classifiers(self,X_train,y_train,X_test, y_test,save_path):
        print('Training using Tpot')
        pipeline_optimizer = TPOTClassifier(generations=5, population_size=50, cv=3,
                                            random_state=0, verbosity=2,scoring='balanced_accuracy')
        pipeline_optimizer.fit(X_train, y_train)
        pipeline_optimizer.export(save_path+'.py')
        print(pipeline_optimizer.score(X_test, y_test))
