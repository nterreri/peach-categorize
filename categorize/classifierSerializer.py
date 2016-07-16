'''
Module responsible for serializing and writing the classifier to file.
'''
import pickle

def serialize(classifier, filepath="categorize/data/classifier.pickle"):
    '''
    Method that serializes a classifier into a .pickle file at the specified
    filepath
    '''
    with open(filepath, "wb") as ostream:
        pickle.dump(classifier, ostream)
