from collections import namedtuple, Iterable, Mapping, UserDict
from fuzzywuzzy import fuzz, process


def Args(**kwargs):
    """Wrapper to easily create a named tuple of arguments. Functions sometimes
    return multiple values, and we have a few options to handle this: we can
    return them as a regular tuple, but it is often convenient to be able to
    reference items by name rather than position. If we want the output to be
    mutable, we can return a dictionary, but this still requires more space
    than a tuple and bracket notation is arguably less convenient than dot
    notation. We can create a new namedtuple inside the function, but this
    kind of seems like overkill to create a new type of namedtuple for each
    function.

    Instead, this lets us create a namedtuple of Args on the fly just as easily
    as making a dictionary.

    Parameters
    ----------

    Examples
    --------
    def math_summary(x, y):
        sum_ = x + y
        prod = x * y
        diff = x - y
        quotient = x / y
        return Args(sum=sum_,
                    product=prod,
                    difference=diff,
                    quotient=quotient)

    >>> results = math_summary(4, 2)
    >>> results.product

    8

    >>> results.quotient

    2

    >>> results

    Args(sum=6, product=8, difference=2, quotient=2)
    """
    args = namedtuple('Args', kwargs.keys())
    return args(*kwargs.values())


class FuzzyKeyDict(dict):
    """Dictionary that will try to find similar keys if a key is missing and
    return their corresponding values. This could be useful when working with
    embeddings, where we could try mapping missing words to a combination of
    existing words.

    Examples
    --------
    d = FuzzyKeyDict(limit=3)
    d['dog'] = 0
    d['cat'] = 1
    d['alley cat'] = 2
    d['pig'] = 3
    d['cow'] = 4
    d['cowbell'] = 5
    d['baby cow'] = 6

    # Keys and similarity scores for the most similar keys.
    >>> d.similar_keys('house cat')
    [('alley cat', 56), ('cat', 50), ('cowbell', 25)]

    # "house cat" not in dict so we get the values for the most similar keys.
    >>> d['house cat']
    [2, 1, 5]

    # "cat" is in dict so output is an integer rather than a list.
    >>> d['cat']
    1
    """

    def __init__(self, data=(), limit=3, return_list=False):
        """
        Parameters
        ----------
        data: Iterable (optional)
            Sequence of pairs, such as a dictionary or a list of tuples. If
            provided, this will be used to populate the FuzzyKeyDict.
        limit: int
            Number of similar keys to find when trying to retrieve the value
            for a missing key.
        return_list: bool
            If True, __getitem__ will always return a list of len `limit`. If
            False, it will return a key's corresponding value if it's present
            and a list of values for the `limit` closest keys if it's not.
        """
        super().__init__(data)
        self.limit = limit

    def __getitem__(self, key):
        """
        Returns
        -------
        any or list[any]: If key is present in dict, the corresponding value
            is returned. If not, the n closest keys are identified and their
            corresponding values are returned in a list (where n is defined
            by the `limit` argument specified in the constructor). Values are
            sorted in descending order by the neighboring keys' similarity to
            the missing key in.
        """
        try:
            res = super().__getitem__(key)
            return [res]*self.limit if self.return_list else res
        except KeyError:
            # super().__getitem__ fails for some reason in this case.
            return [dict.__getitem__(self, k) for k in self.similar_keys(key)]

    def similar_keys(self, key, return_similarities=False):
        """Find the keys in the dictionary that are most similar to the given
        key.

        Parameters
        ----------
        key: str
            This can be present or missing from the dictionary, though in
            practice it's often more useful when it's missing. We'll search
            the existing keys and find the strings that are most similar.
        return_similarities: bool
            If True, return a list of tuples where the first item is a key and
            the second item is its similarity to the given key (higher means
            more similar).

        Returns
        -------
        list: Either a list of strings or a list of tuples depending on
            `return_similarities`.
        """
        pairs = process.extract(key, self.keys(), limit=self.limit,
                                scorer=fuzz.ratio)
        if return_similarities:
            return pairs
        return [p[0] for p in pairs]


class DotDict(dict):
    """Dictionary that allows use of dot notation as well as bracket notation.
    """

    def __getattr__(self, k):
        return self[k]

    def __setattr__(self, k, v):
        self[k] = v

    def __delattr__(self, k):
        del self[k]


class LambdaDict(UserDict):
    """Create a default dict where the default function can accept parameters.
    Whereas the defaultdict in Collections can set the default as int or list,
    here we can pass in any function where the key is the parameter.
    """

    def __init__(self, default_function):
        """
        Parameters
        ----------
        default_function: function
            When referencing a key in a LambdaDict object that has not been
            added yet, the value will be the output of this function called
            with the key passed in as an argument.
        """
        super().__init__()
        self.f = default_function

    def __missing__(self, key):
        self[key] = self.f(key)
        return self[key]
