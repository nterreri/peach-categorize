from categorize.training import train
from categorize.classifierSerializer import serialize
from categorize.classifierDeserializer import deserialize

def main(args):
    classifier = train()
    serialize(classifier)
    classifier = deserialize()
    return classifier

if __name__ == '__main__':
    import sys
    main(sys.argv)
