import re

class RandomStringClassifier():
    def __init__(self):
        self.regex = re.compile('^(?=.*[0-9])([a-zA-Z0-9-.,+/ ]+)$')

    def classify(self,word):
        if re.match(self.regex,word):
            return 1
        else:
            return 0
