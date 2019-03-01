from gensim.models import FastText
import pandas as pd
from sklearn.model_selection import train_test_split
from data.data_constructor import get_company_data,get_items_cat,get_location_data
import numpy as np
#import fasttext

class FeatureBuilder:

    def __init__(self, ordering = ['company','location','goods']):
        self.feature_encoder = None
        self.sizes = []
        self.train = None
        self.validation = None
        self.ordering = ordering
        self.word_mapping = {}

    def load_model(self):
        try:
            self.feature_encoder = FastText.load('./model/fasttext.model')
        except:
            self.load_data()
            self.train_fasttext_encoder()
            self.validate_encoder()

    def load_data(self):
        data = []
        datasets = [get_company_data(),get_location_data(),get_items_cat()]
        for idx, dataset in enumerate(datasets):
            print(dataset.head())
            print('Is Null any entry?:',dataset.isnull().values.any())
            for idx2, row in dataset.iterrows():
                if row['name'] not in self.word_mapping:
                    self.word_mapping[row['name']] = []
                self.word_mapping[row['name']].append(self.ordering[idx])
            self.sizes.append(dataset.shape[0])
            data += list(dataset['name'].values)

        self.train, self.validation = train_test_split(data,random_state=0,test_size=0.2,shuffle=True)
        print('Train Test Constructed')

    def train_fasttext_encoder(self):
        train_sentences = []

        for word in self.train:
            mappings = self.word_mapping[word]
            for mapping in mappings:
                sentence = [word,mapping]
            train_sentences.append(sentence)

        self.feature_encoder = FastText(size=50, window=1, min_count=1,min_n=1,max_n=6)
        self.feature_encoder.build_vocab(sentences=train_sentences)
        self.feature_encoder.train(sentences=train_sentences, total_examples=self.feature_encoder.corpus_count, epochs=1000)
        self.feature_encoder.save('./models/fasttext.model')

        #self.feature_encoder = FastText(size=25, window=1, min_count=1, sentences=train_sentences, iter=50)

    def validate_encoder(self):
        test_words = self.validation
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

featureBuilder = FeatureBuilder()
featureBuilder.load_model()
