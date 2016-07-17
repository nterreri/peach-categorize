'''
Module responsible for splitting a laballed dataset into train, dev and test
sets.
'''
import random

def split(labelled_data, testsizepercent = 0.1, devsizepercent = 0.1, \
          shuffle = False):
    '''
    Returns a triple of train_set, dev_set and test_set for the parameter
    labelled_data set.

    Shuffles the set before splitting if the shuffle flag is set to True.
    '''

    if shuffle:
        random.shuffle(labelled_data)

    testsize = int(len(labelled_data) * testsizepercent)
    devsize = int(len(labelled_data) * devsizepercent)

    test_set  = labelled_data[:testsize]
    dev_set   = labelled_data[testsize:testsize + devsize]
    train_set = labelled_data[testsize + devsize:]

    return train_set, dev_set, test_set
