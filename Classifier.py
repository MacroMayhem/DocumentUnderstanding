import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import pickle
from tpot import TPOTClassifier

import tpot

class Classifier:
    def load_models(self,path):
        try:
            self.svc = pickle.load(open(path+'svm.sav','rb'))
            self.rf = pickle.load(open(path+'rf.sav','rb'))
            self.xgb = pickle.load(open(path+'xgb.sav','rb'))
            self.knn = pickle.load(open(path+'knn.sav','rb'))
        except:
            raise Exception('Missing saved files in path. Train classifiers first')

    def train_classifiers(self,X_train,y_train,X_test, y_test,save_path):
            print('Training...')

            print('SVM')
            self.svc = SVC(gamma='auto')
            self.svc.fit(X_train,y_train)

            print('Random Forest')
            self.rf = RandomForestClassifier(n_estimators=150,max_depth=5,random_state=0)
            self.rf.fit(X_train,y_train)

            print('Gradient Boosting')
            self.xgb = GradientBoostingClassifier(n_estimators=150,max_depth=5,random_state=0)
            self.xgb.fit(X_train,y_train)

            print('K Nearest Neighbours')
            self.knn = KNeighborsClassifier(n_neighbors=7)
            self.knn.fit(X_train,y_train)

            print('Test accuracies')
            print('SVC:',self.svc.score(X_test,y_test))
            print('RF:',self.rf.score(X_test,y_test))
            print('XGB:',self.xgb.score(X_test,y_test))
            print('KNN:',self.knn.score(X_test,y_test))

            pickle.dump(self.svc, open(save_path+'svc.sav', 'wb'))
            pickle.dump(self.rf, open(save_path+'rf.sav', 'wb'))
            pickle.dump(self.xgb, open(save_path+'xgb.sav', 'wb'))
            pickle.dump(self.knn, open(save_path+'knn.sav', 'wb'))

    def tpot_classifiers(self,X_train,y_train,X_test, y_test,save_path):
        print('Training using Tpot')
        pipeline_optimizer = TPOTClassifier(generations=5, population_size=20, cv=3,
                                            random_state=0, verbosity=2,scoring='balanced_accuracy')
        pipeline_optimizer.fit(X_train, y_train)
        pipeline_optimizer.export(save_path+'tpot.py')
        print(pipeline_optimizer.score(X_test, y_test))
