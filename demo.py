import pprint
from categorize.training import train
from categorize.classifierSerializer import serialize
from categorize.classifierDeserializer import deserialize
from categorize.dataset_reading import read_data
from categorize.dataset_splitting import split
from categorize import develop

DATAFILE="categorize/data/sample_data.csv"
FORMAT="csv"
OUTPUTFILE = "categorize/data/classifier.pickle"

def demo(args):

    with open(DATAFILE, 'r') as istream:
        print "Reading data from \"{}\" ...".format(DATAFILE)
        dataset = read_data(istream, format=FORMAT)

    print "Splitting data set ..."
    train_set, dev_set, test_set = split(dataset)

    print "Training classifier ..."
    classifier = train(train_set) # Must become more modular!
    print "Finished training."

    print "Storing classifier in \"{}\"".format(OUTPUTFILE)
    serialize(classifier, filepath=OUTPUTFILE)

    print "Testing classifier against dev set ..."
    develop.test(classifier, dev_set)
    print "Test result:"
    develop.print_errors()

    # classifier = deserialize()
    # return classifier

if __name__ == '__main__':
    import sys
    demo(sys.argv)
