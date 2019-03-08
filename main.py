from Classifier import Classifier
from FeatureBuilder import  FeatureBuilder
import json
from RandomStringClassifier import RandomStringClassifier
from DateClassifier import DateClassifier
import operator

features = FeatureBuilder()
features.load_model()

classifier_3types = Classifier(features.company_feature_encoder,features.location_feature_encoder,features.goods_feature_encoder)
classifier_3types.load_classifiers()

dateClassifier = DateClassifier()
randomClassifier = RandomStringClassifier()

fname = 'test5.json'

with open('./data/'+fname) as f:
    data = json.load(f)

output = {}
confidence_thresh = 0.5

for cats in data['recognitionResult']['lines']:
    for word in cats['words']:
        text = word['text'].lower()
        clean_text = ''.join(c for c in text if c.isalnum())
        is_date = dateClassifier.classify(text)
        is_randomString = randomClassifier.classify(word['text'])
        scores, assignments = classifier_3types.one_vs_all_classification(clean_text, features.ordering)
        class_assigned, mx_score = max(scores.items(), key=operator.itemgetter(1))

        if is_date:
            output[word['text']] = 'Date'
        elif is_randomString:
            output[word['text']] = 'RandomString'
        elif mx_score > confidence_thresh:
            output[word['text']] = class_assigned
        else:
            output[word['text']] = 'Other'

with open('./output/'+fname, 'w') as outfile:
    json.dump(output, outfile,indent=4)