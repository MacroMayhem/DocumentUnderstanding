from gensim.models import FastText
import pandas as pd
from sklearn.model_selection import train_test_split
from data.data_constructor import get_company_data,get_items_cat,get_location_data
import numpy as np
from sklearn.utils import  shuffle

class FeatureBuilder:

    def __init__(self, ordering = ['company','location','goods']):
        self.feature_encoder = None
        self.sizes = []
        self.train = None
        self.validation = None
        self.ordering = ordering
        self.word_mapping = {}
        self.company_feature_encoder = None
        self.location_feature_encoder = None
        self.goods_feature_encoder = None

    def load_data(self):
        self.load_data()

    def load_model(self):
        try:
            #self.feature_encoder = FastText.load('./models/fasttext.model')
            self.company_feature_encoder = FastText.load('./models/company_fasttext.model')
            self.location_feature_encoder = FastText.load('./models/location_fasttext.model')
            self.goods_feature_encoder = FastText.load('./models/goods_fasttext.model')
        except:
            print('Existing model does not exist. Training from scratch')
            self.classType_fasttext_train('company')
            self.classType_fasttext_train('location')
            self.classType_fasttext_train('goods')
            #self.train_fasttext_encoder()
            #self.validate_encoder()

    def load_data(self):
        data = []
        datasets = [get_company_data(),get_location_data(),get_items_cat()]
        for idx, dataset in enumerate(datasets):
            print('Is any entry Null?:',dataset.isnull().values.any())
            for idx2, row in dataset.iterrows():
                if row['name'] not in self.word_mapping:
                    self.word_mapping[row['name']] = []
                self.word_mapping[row['name']].append(self.ordering[idx])
            self.sizes.append(dataset.shape[0])
            data += list(dataset['name'].values)
        data = shuffle(data,random_state=0)
        self.train, self.validation = train_test_split(data,random_state=0,test_size=0.2)
        print('Train Test Constructed')


    def classType_fasttext_train(self,classType):

        train_sentences = []

        for word in self.train:
            sentence = []
            mappings = self.word_mapping[word]
            for mapping in mappings:
                if mapping == classType:
                    sentence.append(word)
            if len(sentence) > 0:
                train_sentences.append(sentence)

        feature_encoder = FastText(size=50, window=2, min_count=1,min_n=2,max_n=6)
        feature_encoder.build_vocab(sentences=train_sentences)
        feature_encoder.train(sentences=train_sentences, total_examples=feature_encoder.corpus_count, epochs=1000)
        feature_encoder.save('./models/'+classType+'_fasttext.model')
        if classType == 'company':
            self.company_feature_encoder = feature_encoder
        elif classType == 'location':
            self.location_feature_encoder = feature_encoder
        elif classType == 'goods':
            self.goods_feature_encoder = feature_encoder
        else:
            raise Exception('Allowed arguments are company, location and goods')
        #self.feature_encoder = FastText(size=25, window=1, min_count=1, sentences=train_sentences, iter=50)


    def train_fasttext_encoder(self):
        train_sentences = []

        for word in self.train:
            mappings = self.word_mapping[word]
            for mapping in mappings:
                sentence = [word]
            train_sentences.append(sentence)

        self.feature_encoder = FastText(size=50, window=2, min_count=1,min_n=2,max_n=6)
        self.feature_encoder.build_vocab(sentences=train_sentences)
        self.feature_encoder.train(sentences=train_sentences, total_examples=self.feature_encoder.corpus_count, epochs=1000)
        self.feature_encoder.save('./models/fasttext.model')

        #self.feature_encoder = FastText(size=25, window=1, min_count=1, sentences=train_sentences, iter=50)

    def validate_encoder(self):
        test_words = self.validation

        ## Finding the closest cluster center (Company, Location or Good)
        tp = 0


        for word in test_words:
            distances = []
            encoding = self.feature_encoder[word]
            for order in self.ordering:
                category_encoding = self.feature_encoder[order]
                distances.append(np.linalg.norm(encoding-category_encoding))
            idx = distances.index(min(distances))

            gt_categories = self.word_mapping[word]
            for gt_category in gt_categories:
                if self.ordering[idx] == gt_category:
                    tp+= 1
                    break

        print('Closest cluster center validation approach accuracy:',str(tp/len(test_words)))

        ## Doing the K-nearest analysis
        tp = 0

        order_idx = {}
        for idx,order in enumerate(self.ordering):
            order_idx[order] = idx

        for word in test_words:
            distances = []
            encoding = self.feature_encoder[word]
            nearest_neighbours = self.feature_encoder.most_similar(word,topn=15)
            votes = [0,0,0]

            for neighbour in nearest_neighbours:
                mappings = self.word_mapping[neighbour[0]]
                for mapping in mappings:
                    votes[order_idx[mapping]]+=1

            assigned_idx = votes.index(max(votes))

            gt_categories = self.word_mapping[word]
            for gt_category in gt_categories:
                if self.ordering[assigned_idx] == gt_category:
                    tp += 1
                    break

        print('Nearest 15-Neighbour accuracy:', str(tp / len(test_words)))


    def one_vs_rest_generator(self,positive_index=None):

        assert positive_index is not None, "Requires index for the positive class(see ordering)"
        if self.ordering[positive_index]  == 'company':
            feature_encoder = self.company_feature_encoder
        elif self.ordering[positive_index]  == 'location':
            feature_encoder = self.location_feature_encoder
        elif self.ordering[positive_index]  == 'goods':
            feature_encoder = self.location_feature_encoder
        else:
            raise Exception('Marked positive class not in the set {0,1,2}')

        X_train = []
        y_train = []
        X_test = []
        y_test = []

        for word in self.train:
            try:
                X_train.append(feature_encoder[word])
                if self.ordering[positive_index] in self.word_mapping[word]:
                    y_train.append(1)
                else:
                    y_train.append(0)
            except KeyError:
                print('all ngrams for word %s absent from model. Skipping for %s'% (word,self.ordering[positive_index]))


        for word in self.validation:
            try:
                X_test.append(feature_encoder[word])
                if self.ordering[positive_index] in self.word_mapping[word]:
                    y_test.append(1)
                else:
                    y_test.append(0)
            except KeyError:
                print('all ngrams for word %s absent from model. Skipping for %s' % (word, self.ordering[positive_index]))

        return np.asarray(X_train,dtype=np.float64),np.asarray(y_train,dtype=np.float64),\
               np.asarray(X_test,dtype=np.float64),np.asarray(y_test,dtype=np.float64)


    def get_encoding(self,word):
        return self.feature_encoder[word]