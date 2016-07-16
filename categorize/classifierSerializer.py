import pickle

def serialize(classifier, filepath="categorize/data/classifier.pickle"):
    with open(filepath, "wb") as ostream:
        pickle.dump(classifier, ostream)
