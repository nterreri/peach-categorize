import pprint
from categorize.training import train
from categorize.classifierSerializer import serialize
from categorize.classifierDeserializer import deserialize
from categorize.dataset_reading import read_data

datafilepath="categorize/data/sample_data.csv"

def demo(args):
    with open(datafilepath, 'r') as istream:
        labelled_data = read_data(istream, format="csv")

    # pprint.pprint(labelled_data)
    testsize = int(len(labelled_data) * 0.1)
    # print "Testsize: {}".format(testsize)
    devsize = int((len(labelled_data) - testsize) * 0.1 )
    # print "Devsize: {}".format(devsize)
    # print "len(labelled_data) - testsize: {}".format(len)
    # print "testsize + devsize: {}".format(testsize + devsize)

    test_set  = labelled_data[:testsize]
    dev_set   = labelled_data[testsize:testsize + devsize]
    train_set = labelled_data[testsize + devsize:]

    print "\n\nTrain set:"
    pprint.pprint(train_set)
    print "\n\nDev set:"
    pprint.pprint(dev_set)
    print "\n\nTest set:"
    pprint.pprint(test_set)

    for pair in train_set:
        assert pair not in dev_set
        assert pair not in test_set

    # classifier = train()
    # serialize(classifier)
    # classifier = deserialize()
    # return classifier

if __name__ == '__main__':
    import sys
    demo(sys.argv)
