
from Classifier import Classifier
from FeatureBuilder import  FeatureBuilder



features = FeatureBuilder()
features.load_model()

company_classifier_path = './models/one_vs_rest_company'
location_classifier_path = './models/one_vs_rest_location'
goods_classifier_path = './models/one_vs_rest_goods'

company_X_train, company_y_train,company_X_test, company_y_test = features.one_vs_rest_generator(0)
#location_X_train, location_y_train,location_X_test, location_y_test = features.one_vs_rest_generator(1)
#goods_X_train, goods_y_train, goods_X_test, goods_y_test = features.one_vs_rest_generator(2)

classifier = Classifier()
classifier.train_classifiers(company_X_train,company_y_train,company_X_test,company_y_test,company_classifier_path)
classifier.tpot_classifiers(company_X_train,company_y_train,company_X_test,company_y_test,company_classifier_path)