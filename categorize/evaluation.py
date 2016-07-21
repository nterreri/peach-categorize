'''
Evaluates a classifier against a test set.
'''
import collections
from nltk.metrics import scores

class ClassifierEvaluator:

    # _classifier = None
    # _gold = None
    # _reference_sets = collections.defaultdict(set)
    # _test_sets = collections.defaultdict(set)

    def __init__(self, classifier, gold):
        self._classifier = classifier
        self._gold = gold
        self._reference_sets, self._test_sets = \
                            self._get_reference_and_test_sets(classifier, gold)

    def _get_reference_and_test_sets(self, classifier, test):
        '''
        Gets a reference and test set for the classifier over the specified test
        set.

        See: http://streamhacker.com/2010/05/17/text-classification-sentiment-analysis-precision-recall/
        '''
        reference_sets = collections.defaultdict(set)
        test_sets = collections.defaultdict(set)
        for i, (data_point, label) in enumerate(test):
            reference_sets[label].add(i)
            guess = classifier.classify(data_point)
            test_sets[guess].add(i)

        return reference_sets, test_sets

    def precision(self, label):
        return scores.precision(self._reference_sets[label], \
                                self._test_sets[label])

    def recall(self, label):
        return scores.recall(self._reference_sets[label], \
                            self._test_sets[label])

    def f_measure(self, label, alpha=0.5):
        return scores.f_measure(self._reference_sets[label], \
                                self._test_sets[label], alpha)

    def accuracy(self):
        return self._classifier.accuracy(self._gold)
