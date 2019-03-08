from Classifier import Classifier
from FeatureBuilder import  FeatureBuilder
import operator

features = FeatureBuilder()
features.load_data()
features.load_model()


company_X_train, company_y_train,company_X_test, company_y_test = features.one_vs_rest_generator(0)
location_X_train, location_y_train,location_X_test, location_y_test = features.one_vs_rest_generator(1)
goods_X_train, goods_y_train, goods_X_test, goods_y_test = features.one_vs_rest_generator(2)

classifier = Classifier(features.company_feature_encoder,features.location_feature_encoder,features.goods_feature_encoder)
classifier.load_classifiers()
#classifier.fit_classifier('company',company_X_train,company_y_train)
#classifier.fit_classifier('location',location_X_train,location_y_train)
#classifier.fit_classifier('goods',goods_X_train,goods_y_train)
tp = 0
fp = 0

confidence_thresh = 0.1

for word in features.validation:
    scores,assignments = classifier.one_vs_all_classification(word,features.ordering)
    class_assigned, mx_score = max(scores.items(),key=operator.itemgetter(1))
    mappings = features.word_mapping[word]
    true_p = False

    for mapping in mappings:
        if mapping == class_assigned and mx_score > confidence_thresh:
            tp += 1
            true_p = True
            break

    if true_p is False:
        fp += 1

print('Recall:',tp/len(features.validation))
print(fp/len(features.validation))






