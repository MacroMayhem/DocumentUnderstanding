from tpot import TPOTClassifier
import numpy as np

class Classifier:

    def __init__(self,company_encoder,location_encoder,goods_encoder):
        self.company_classifier = None
        self.location_classifier = None
        self.goods_classifier = None
        self.company_encoder = company_encoder
        self.location_encoder = location_encoder
        self.goods_encoder = goods_encoder

    def tpot_classifiers(self,X_train,y_train,X_test, y_test,save_path):
        print('Training using Tpot')
        pipeline_optimizer = TPOTClassifier(generations=10, population_size=25, cv=3,
                                            random_state=0, verbosity=2,scoring='balanced_accuracy')
        pipeline_optimizer.fit(X_train, y_train)
        pipeline_optimizer.export(save_path+'.py')
        print(pipeline_optimizer.score(X_test, y_test))

    def load_classifiers(self):
        try:
            from models.one_vs_rest_company import CompanyClassifier
            from models.one_vs_rest_location import LocationClassifier
            from models.one_vs_rest_goods import GoodsClassifier

            self.company_classifier = CompanyClassifier()
            self.company_classifier.load_model()
            self.location_classifier = LocationClassifier()
            self.location_classifier.load_model()
            self.goods_classifier = GoodsClassifier()
            self.goods_classifier.load_model()

        except:
            raise ImportError('Trained models not available.')

    def fit_classifier(self,classType,X,Y):
        try:
            from models.one_vs_rest_company import CompanyClassifier
            from models.one_vs_rest_location import LocationClassifier
            from models.one_vs_rest_goods import GoodsClassifier


            if classType == 'company':
                self.company_classifier = CompanyClassifier()
                self.company_classifier.fit(X,Y)
            if classType == 'location':

                self.location_classifier = LocationClassifier()
                self.location_classifier.fit(X,Y)
            if classType == 'goods':

                self.goods_classifier = GoodsClassifier()
                self.goods_classifier.fit(X,Y)
        except:
            raise ImportError('Trained models not available.')
    def one_vs_all_classification(self,word,orderings):

        scores = {}
        assignments = {}

        for ordering in orderings:
            scores[ordering] = 0
            assignments[ordering] = 0
            if ordering == 'company':
                encoder = self.company_encoder
                classifier = self.company_classifier
            elif ordering == 'location':
                encoder = self.location_encoder
                classifier = self.location_classifier
            elif ordering == 'goods':
                encoder = self.goods_encoder
                classifier = self.goods_classifier
            else:
                raise Exception('Invalid classtype')
            try:
                word_feature = np.reshape(encoder[word],(1,-1))
                probability = classifier.probability(word_feature)
                scores[ordering] = probability[0][1]
                assignments[ordering] = classifier.predict(word_feature)
            except:
                print('%s does not exist in the encoder model for %s class type'%(word,ordering))
                scores[ordering] = 0
                assignments[ordering] = 0
        return scores, assignments
