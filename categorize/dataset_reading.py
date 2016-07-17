'''
Module responsible for reading a dataset from file.

'''
from textblob import formats as formats

def read_data(dataset, format=None, **kwargs):
    """Reads a data file and returns an iterable that can be used
    as testing or training data.

    Adapted from:
    https://github.com/sloria/TextBlob/blob/dev/textblob/classifiers.py#L128
    """
    # Attempt to detect file format if "format" isn't specified
    if not format:
        format_class = formats.detect(dataset)
        if not format_class:
            raise FormatError('Could not automatically detect format for the '
                              'given data source.')
    else:
        registry = formats.get_registry()
        if format not in registry.keys():
            raise ValueError("'{0}' format not supported.".format(format))
        format_class = registry[format]
    return format_class(dataset, **kwargs).to_iterable()
