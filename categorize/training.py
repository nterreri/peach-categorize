'''
Module responsible for the training of a categorizer.

'''
import textblob.classifiers

def train(datafilepath="categorize/data/sample_data.csv"):
    with open(datafilepath, 'r') as istream:
        classifier = textblob.classifiers.NaiveBayesClassifier(istream, format="csv")
    return classifier
