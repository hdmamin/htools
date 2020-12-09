from collections import namedtuple, UserDict
from datasketch import MinHash, MinHashLSHForest
from functools import partial
from fuzzywuzzy import fuzz, process
import numpy as np
import warnings

from htools.core import ngrams
from htools.meta import add_docstring


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


class _FuzzyDictBase(dict):
    """Abstract base class for fuzzy key dictionaries.
    Subclasses must define a method `similar` (see FuzzyKeyDict and LSHDict for
    examples. See `__init_subclass__` for explanation of why we don't define
    an abstractmethod here.
    """

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError:
            return self.similar(key, mode='values', n_keys=1)[0]

    def __init_subclass__(cls, **kwargs):
        """Can't use abstractmethod decorator because we're inheriting from
        dict whose metaclass is not ABCMeta. Enforce this manually instead.
        """
        super().__init_subclass__(**kwargs)
        # Don't use inspect.ismethod because functions haven't been bound yet.
        # Not perfect but good enough for this use case.
        if not callable(getattr(cls, 'similar', None)):
            raise TypeError('Children of _FuzzyDictBase must define method '
                            '`similar`.')

    def _filter_similarity_pairs(self, pairs, mode='keys_values'):
        """
        mode: str
            Determines what information to return. In each case, items are
            sorted in descending order of similarity.
            - "keys_values": list of (key, value) tuples.
            - "keys_similarities": list of (key, similarity) tuples.
            - "keys_values_similarities": list of (key, value, similarity)
                tuples.
            - "keys": list of keys (strings).
            - "values": list of values corresponding to the nearest keys (type
                is dependent on what your dict values are).
        """
        if mode == 'values':
            return [self[p[0]] for p in pairs]
        elif mode == 'keys_values':
            return [(p[0], self[p[0]]) for p in pairs]
        elif mode == 'keys_similarities':
            return pairs
        elif mode == 'keys':
            return [p[0] for p in pairs]
        elif mode == 'keys_values_similarities':
            return [(p[0], self[p[0]], p[1]) for p in pairs]
        else:
            raise ValueError(
                'Unrecognized value for `mode`. Should be one of ("values", '
                '"keys", keys_values", "keys_similarities", or '
                '"keys_values_similarities").'
            )


class LSHDict(_FuzzyDictBase):
    """Dictionary that returns the value corresponding to a key's nearest
    neighbor if the key isn't present in the dict. This is intended for use
    as a word2index dict when using embeddings in deep learning: e.g. if we
    have domain embeddings for the top 100k websites, some of our options for
    dealing with unknown domains are:

    1. Encode all of them as <UNK>. This loses a lot of information.
    2. Create a FuzzyKeyDict which will search for similar keys using variants
    of Levenshtein distance. Lookup is O(N) and for 100k domains, that comes
    out to 0.6 seconds per item. We might have thousands or millions of
    lookups over the course of training so this can be a significant cost.
    3. Create an LSHDict (lookups are O(1)). Indexing into the dict as usual
    (e.g. my_lsh_dict[key]) will provide the key's index if present and the
    (approximate) nearest neighbor's index otherwise. Either way, the result
    can be used to index into your embedding layer.
    4. Create an LSHDict and use the `similar_values` method to return n>1
    neighbors. Then pass their indices to an Embedding layer and
    compute the sum/average/weighted average of the results. This may be
    preferable to #3 cases such as web domain lookup, where similar URLs are
    not guaranteed to represent similar sites. (This is basically
    equivalent to an EmbeddingBag layer, but in torch that doesn't store
    intermediate representations so we wouldn't be able to use our pretrained
    embeddings.)
    """

    def __init__(self, data, n_candidates=None, n_keys=3, ngram_size=3,
                 scorer=fuzz.ratio):
        """
        Parameters
        ----------
        data: dict or list[tuple]
            The base dictionary. Unlike FuzzyKeyDict, we require this since
            adding items one by one is computationally infeasible for large
            datasets. Just build up your dictionary first.
        n_candidates: int or None
            Number of reasonably similar keys to retrieve when trying to index
            in with a key that's missing (or when using the `similar` method).
            You can override this in `similar` but not when using
            __getitem__'s square bracket syntax. If not specified, this will
            be auto initialized to vocab size/1,000, clipped to lie in
            [20, 500]. See `similar` docstring for more on this.
        n_keys: int
            Default number of similar keys to retrieve in `similar`.
        scorer: function
            Default scoring function to use to narrow `n_candidates` keys down
            to `n_keys`. Should be a fuzzywuzzy function where scores lie in
            [0, 100] and higher values indicate high similarity.
        """
        if len(data) < 10_000 and len(next(iter(data))) < 100:
            warnings.warn(
                'It looks like you\'re working with a relatively small '
                'amount of data. FuzzyKeyDict may be fast enough for your '
                'use case and would provide the set of strictly most similar '
                'keys rather than an approximation of that set.'
            )

        super().__init__(data)
        self.scorer = scorer
        self.hash_word = partial(self.lsh_hash_word, n=ngram_size)
        self.forest = MinHashLSHForest(num_perm=128)
        self._initialize_forest()

        # Datasketch's LSH implementation usually gives pretty decent results
        # even with numbers as low as 5-10, but increasing that by a factor of
        # 10 comes with minimal time cost: Fuzzywuzzy matching doesn't get
        # particularly slow until we get into the thousands. The fact that
        # we cap this at 500 makes this lookup asymptotically O(1) while
        # FuzzyKeyDict's is O(N).
        self.n_candidates = n_candidates or np.clip(len(self) // 1_000,
                                                    20, 500)
        self.n_keys = n_keys

    def __setitem__(self, key, val):
        """Try to add keys all at once in the constructor because adding new
        keys can be extremely slow.
        """
        super().__setitem__(key, val)
        self._update_forest(key, val)

    def _update_forest(self, key, val, index=True):
        """Used in __setitem__ to update our LSH Forest. Forest's index method
        seems to recompute everything so adding items to a large LSHDict will
        be incredibly slow. Luckily, our deep learning use case rarely/never
        requires us to update object2index dicts after instantiation so that's
        not as troubling as it might seem.

        Parameters
        ----------
        key: str
        val: any
        index: bool
            If True, reindex the forest (essentially making the key
            queryable). This should be False when initializing the forest so
            we just index once after everything's been added.
        """
        self.forest.add(key, self.hash_word(key))
        if index: self.forest.index()

    def _initialize_forest(self):
        """Called once in __init__ to add all items to LSH Forest. This is
        necessary because dict specifically calls its own __setitem__, not
        its children's.
        """
        for k, v in self.items():
            self._update_forest(k, v, False)
        self.forest.index()

    @add_docstring(_FuzzyDictBase._filter_similarity_pairs)
    def similar(self, key, mode='keys_values', n_candidates=None,
                n_keys=None, scorer=None):
        """Find a list of similar keys. This is used in __getitem__ but can
        also be useful as a user-facing method if you want to get more than
        1 neighbor or you want to get similarity scores as well.

        Parameters
        ----------
        key: str
            Word/URL/etc. to find similar keys to.
        mode: str
            See section below `Returns`.
        n_candidates: int or None
            Number of similar candidats to retrieve. This uses Jaccard
            Similarity which isn't always a great metric for string
            similarity. This is also where the LSH comes in so they're not
            strictly the n best candidates, but rather a close approximation
            of that set. If None, this will fall back to self.n_candidates.
            Keep in mind this determines how many keys to
        n_keys: int or None
            Number of similar keys to return. If None, this will fall back to
            self.n_keys.
        scorer: function or None
            Fuzzywuzzy scoring function, e.g. fuzz.ratio or
            fuzz.partial_ratio, which will be used to score each candidate and
            select which to return. Higher scores indicate higher levels of
            similarity. If None, this will fall back to self.scorer.

        Returns
        -------
        list: List if `mode` is "keys" or "values". List of tuples otherwise.
        """
        candidates = self.forest.query(self.hash_word(key),
                                       n_candidates or self.n_candidates)
        if not candidates: raise KeyError('No similar keys found.')

        # List of (key, score) where higher means more similar.
        pairs = process.extract(key, candidates,
                                limit=n_keys or self.n_keys,
                                scorer=scorer or self.scorer)
        return self._filter_similarity_pairs(pairs, mode=mode)

    @staticmethod
    @add_docstring(ngrams)
    def lsh_hash_word(word, num_perm=128, **ngram_kwargs):
        """Hash an input word (str) and return a MinHash object that can be
        added to an LSHForest.

        Parameters
        ----------
        word: str
            Word to hash.
        num_perm: int
        ngram_kwargs: any
            Forwarded to `ngrams`.

        Returns
        -------
        datasketch MinHash object
        """
        mhash = MinHash(num_perm=num_perm)
        for ng in ngrams(word, **ngram_kwargs):
            mhash.update(ng.encode('utf8'))
        return mhash


class FuzzyKeyDict(_FuzzyDictBase):
    """Dictionary that will try to find the most similar key if a key is
    missing and return its corresponding value. This could be useful when
    working with embeddings, where we could map missing items to the indices
    of one or more existing embeddings.

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
    >>> d.similar('house cat', mode='keys_similarities')
    [('alley cat', 56), ('cat', 50), ('cowbell', 25)]

    # "house cat" not in dict so we get the value for the most similar key.
    >>> d['house cat']
    2

    # "cat" is in dict so __getitem__ functions like a regular dict.
    >>> d['cat']
    1
    """

    def __init__(self, data=(), n_keys=3, scorer=fuzz.ratio):
        """
        Parameters
        ----------
        data: Iterable (optional)
            Sequence of pairs, such as a dictionary or a list of tuples. If
            provided, this will be used to populate the FuzzyKeyDict.
        n_keys: int
            Default number of similar keys to retrieve when using the
            `similar` method.
        scorer: function
            Default scoring function to use to select the most similar keys in
            `similar`. Should be a fuzzywuzzy function where scores lie in
            [0, 100] and higher values indicate high similarity.
        """
        super().__init__(data)
        self.n_keys = n_keys
        self.scorer = scorer

    @add_docstring(_FuzzyDictBase._filter_similarity_pairs)
    def similar(self, key, mode='keys_values', n_keys=None, scorer=None):
        """Find the keys in the dictionary that are most similar to the given
        key and return some relevant information (the keys, their similarity
        scores, their corresponding values, etc.) depending on what the user
        specifies.

        Parameters
        ----------
        key: str
            This can be present or missing from the dictionary, though in
            practice it's often more useful when it's missing. We'll search
            the existing keys and find the strings that are most similar.
        mode: str
            See section below `Returns`.
        n_keys: int or None
            Number of similar keys to return. If None, this will fall back to
            self.n_keys.
        scorer: function or None
            Fuzzywuzzy scoring function, e.g. fuzz.ratio or
            fuzz.partial_ratio, which will be used to score each candidate and
            select which to return. Higher scores indicate higher levels of
            similarity. If None, this will fall back to self.scorer.

        Returns
        -------
        list: List if `mode` is "keys" or "values". List of tuples otherwise.
        """
        pairs = process.extract(key,
                                self.keys(),
                                limit=n_keys or self.n_keys,
                                scorer=scorer or fuzz.ratio)
        return self._filter_similarity_pairs(pairs, mode=mode)


class DotDict(dict):
    """Dictionary that allows use of dot notation as well as bracket notation.
    This should be picklable starting in htools>=6.0.6.
    """

    def __getattr__(self, k):
        return self[k]

    def __setattr__(self, k, v):
        self[k] = v

    def __delattr__(self, k):
        del self[k]

    def __getstate__(self):
        """We often have to use `__reduce__` to make dict subclasses picklable
        but this seems to work in this case, I think because we don't have any
        required arguments to __init__.
        """
        return dict(self)

    def __setstate__(self, data):
        self.update(data)


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
