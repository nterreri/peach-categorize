from categorize.evaluation import ClassifierEvaluator
from utils import mock_classifier

LABELS = ["pos","neg"]

def test_init():
    classifier = mock_classifier.get_mock()
    evaluator = ClassifierEvaluator(classifier, mock_classifier.test)

def test_accuracy():
    classifier = mock_classifier.get_mock()
    evaluator = ClassifierEvaluator(classifier, mock_classifier.test)
    accuracy = evaluator.accuracy()

    assert isinstance(accuracy, float) #or accuracy is None

def test_macroavgprecision():
    classifier = mock_classifier.get_mock()
    evaluator = ClassifierEvaluator(classifier, mock_classifier.test)
    avgPrecision = evaluator.macroAvgPrecision(LABELS)

    assert avgPrecision is not None
    assert _isProbabilityMeasure(avgPrecision)
    assert 0.875 == avgPrecision

def test_macroavgrecall():
    classifier = mock_classifier.get_mock()
    evaluator = ClassifierEvaluator(classifier, mock_classifier.test)
    avgRecall = evaluator.macroAvgRecall(LABELS)

    print avgRecall
    assert avgRecall is not None
    assert _isProbabilityMeasure(avgRecall)
    assert 0.833 == round(avgRecall, 3)


def _isProbabilityMeasure(candidate):
    return candidate >= 0 and candidate <= 1

# def test_precision():
#     classifier = mock_classifier.get_mock()
#     evaluator = ClassifierEvaluator(classifier, mock_classifier.test)
#
#     precision = evaluator.precision(' pos')
#     assert isinstance(precision, float)# or precision is None
#
# def test_recall():
#     classifier = mock_classifier.get_mock()
#     evaluator = ClassifierEvaluator(classifier, mock_classifier.test)
#
#     recall = evaluator.recall(' pos')
#     assert isinstance(recall, float)# or recall is None
#
# def test_fmeasure():
#     classifier = mock_classifier.get_mock()
#     evaluator = ClassifierEvaluator(classifier, mock_classifier.test)
#
#     f = evaluator.f_measure(' pos')
#     assert isinstance(f, float)# or f is None
