import pprint
from categorize.training import train
from categorize.classifierSerializer import serialize
from categorize.classifierDeserializer import deserialize
from categorize.dataset_reading import read_data

def demo(args):

    # classifier = train()
    # serialize(classifier)
    # classifier = deserialize()
    # return classifier

if __name__ == '__main__':
    import sys
    demo(sys.argv)
