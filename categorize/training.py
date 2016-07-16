'''
Module responsible for the training of a categorizer.

'''
import textblob.classifiers

def train(datafilepath="categorize/data/sample_data.csv"):
    '''
    Trains a classifier over the data at the filepath.
    Assumes file format is csv.
    '''
    with open(datafilepath, 'r') as istream:
        classifier = textblob.classifiers.NaiveBayesClassifier(istream, format="csv")
    return classifier
