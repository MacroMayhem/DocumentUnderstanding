from dateutil.parser import parse

class DateClassifier():
    def classify(self,word):
        try:
            parse(word)
            return 1
        except ValueError:
            return 0