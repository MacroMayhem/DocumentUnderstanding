from DateClassifier import DateClassifier
from RandomStringClassifier import RandomStringClassifier

dateClassifier = DateClassifier()
randomStringClassifier = RandomStringClassifier()

print(dateClassifier.classify("Dec 25th 2016"))
print(dateClassifier.classify("5.1.2010"))
print(randomStringClassifier.classify('ABCDE 123A'))