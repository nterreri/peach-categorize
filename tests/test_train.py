import pytest
import categorize.training
from textblob.classifiers import BaseClassifier

def test_good():
    classifier = categorize.training.train()
    assert isinstance(classifier, BaseClassifier)

def test_emptyfilepath():
    with pytest.raises(IOError):
        categorize.training.train("")

def test_nosuchfile():
    with pytest.raises(IOError):
        categorize.training.train("nofileherehopefully")
