import pprint
from categorize.training import train
from categorize.classifierSerializer import serialize
from categorize.classifierDeserializer import deserialize
from categorize.dataset_reading import read_data
from categorize.dataset_splitting import split
from categorize import develop
from categorize.evaluation import ClassifierEvaluator

DATAFILE="categorize/data/sample_data.csv"
FORMAT="csv"
OUTPUTFILE = "categorize/data/classifier.pickle"
LABELS = ["pos", "neg"]
def demo(args):

    with open(DATAFILE, 'r') as istream:
        print "Reading data from \"{}\" ...".format(DATAFILE)
        dataset = read_data(istream, format=FORMAT)

    print "Splitting data set ..."
    train_set, dev_set, test_set = split(dataset, shuffle=True)

    print "Training classifier ..."
    classifier = train(train_set) # Must become more modular!
    print "Finished training."

    print "Storing classifier in \"{}\"".format(OUTPUTFILE)
    serialize(classifier, filepath=OUTPUTFILE)

    print "Testing classifier against dev set ..."
    develop.test(classifier, dev_set)
    print "Test result:"
    develop.print_errors()

    print "Initializing evaluator instance ..."
    evaluator = ClassifierEvaluator(classifier, test_set)

    accuracy_score = evaluator.accuracy()
    print "Accuracy: {:.2f}".format(accuracy_score)

    for label in LABELS:
        recall_score = evaluator.recall(label)
        precision_score = evaluator.precision(label)
        fmeasure_score = evaluator.f_measure(label)

        print "Recall for '{}': {:.2f}".format(label, recall_score)
        print "Precision for '{}': {:.2f}".format(label, precision_score)
        print "F for '{}': {:.2f}".format(label, fmeasure_score)

if __name__ == '__main__':
    import sys
    demo(sys.argv)
