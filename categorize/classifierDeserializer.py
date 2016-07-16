import pickle

def deserialize():
    with open("categorize/data/classifier.pickle", "rb") as istream:
        classifier = pickle.load(istream)
    return classifier
