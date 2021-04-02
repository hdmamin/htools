from collections import namedtuple, OrderedDict
from copy import deepcopy
from datasketch import MinHash, MinHashLSHForest
from functools import partial
from fuzzywuzzy import fuzz, process
from heapq import heappop, heappush
import numpy as np
import warnings

from htools.core import ngrams, tolist, identity, func_name
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


class TrieNode:
    """Single node in a Trie. Most of the functionality is provided in Trie
    class rather than here.
    """

    def __init__(self, data=()):
        """
        Parameters
        ----------
        data: Iterable or Iterable[Iterable]
            One or more sequences to add to the node. This could be a string,
            a list of strings, a list of tuples of ints, etc.
        """
        self.edges = {}
        self.stop_state = False
        for x in tolist(data):
            self.append(x)

    def append(self, seq):
        if not seq:
            self.stop_state = True
            return
        x = seq[0]
        if x not in self.edges:
            self.edges[x] = TrieNode()
        self.edges[x].append(seq[1:])

    def __repr__(self):
        return f'{type(self).__name__}({list(self.edges.keys()) or ""})'


class Trie:
    """Memory-efficient data structure for highly duplicated sequence data. For
    example, this would be a nice way to store a dictionary since many words
    overlap (e.g. can, cannot, cannery, canning, cane, and cannon all share the
    first 3 letters. With Trie, the common prefix is stored only once.).
    Checking if a sequence is present is therefore O(n) where n is the length
    of the input sequence. Notice this is unaffected by the number of values in
    the Trie.
    """

    def __init__(self, values=(), suffix=False):
        """
        Parameters
        ----------
        values: str or list-like Iterable
            If provided, this should be one or more sequences to add to the
            try. Sequences could be strings, lists of strings (like word
            tokens), tuples of integers, etc. As of Dec 2020, this should NOT
            be numpy arrays or torch tensors.
        """
        self.head = TrieNode()
        if suffix:
            self._maybe_reverse = lambda x: x[::-1]
        else:
            self._maybe_reverse = identity
        self.suffix = suffix

        # dtype records the type of object present in the trie, and is a
        # string rather than a type because lolviz library has a
        # bug when displaying type attributes. Its visualizations are very
        # helpful here so I don't want to break compatibility.
        self.dtype = ''
        self.child_dtype = ''
        self.postprocess = None

        # Use extend rather than passing values directly to TrieNode because
        # that won't give us validation or preprocessing.
        self._length = 0
        self.extend(tolist(values))

    def append(self, seq):
        """Add a sequence to the trie. This operates in place."""
        if not self.postprocess:
            self.dtype = type(seq).__name__
            self.child_dtype = type(seq[0]).__name__
            self.postprocess = partial(str.join, '') if self.dtype == 'str' \
                else identity
        else:
            self._validate_input(seq)
        self.head.append(self._maybe_reverse(seq))
        self._length += 1

    def extend(self, seqs):
        """Add a list-like group of sequences to the Trie."""
        for seq in seqs:
            self.append(seq)

    def __add__(self, seq):
        """Allows us to add items to a trie using + operator. This does not
        alter the trie in place: to do that, use `append` or assign the result
        of this method back to your variable.

        Returns
        -------
        Trie
        """
        clone = deepcopy(self)
        clone.append(seq)
        return clone

    def _find(self, seq, node=None):
        """Try to find a a sequence in the trie. We provide this helper method
        rather than doing it entirely in __contains__ in case other methods
        want to make use of the found node (perhaps passing it to
        self._values.)

        Returns
        -------
        TrieNode: If node.stop_state=True, the seq is in the trie. If False,
        it's not.
        """
        self._validate_input(seq)
        seq = self._maybe_reverse(seq)
        node = node or self.head
        for x in seq:
            if x not in node.edges:
                # Return this so __contains__ can check its stop state.
                return TrieNode()
            node = node.edges[x]
        return node

    def __contains__(self, seq):
        """Check if a sequence is present in the trie.

        Returns
        -------
        bool
        """
        return self._find(seq).stop_state

    def _values(self, current=None, node=None):
        """Generator that yields each sequence in the tree one by one. Don't
        rely on the order, but I believe it should be a depth first traversal
        where the order of subtries traversed is determined by insertion
        order. See examples.

        Parameters
        ----------
        current: list or None
            List of partial sequence currently being retrieved. This is used
            internally but should rarely need to be called by the user.
        node: TrieNode or None
            The node to retrieve values from. By default, we use the root
            node, thereby retrieving values for the whole trie.

        Examples
        --------
        >>> t = Trie(['add', 'subtract', 'addition', 'multiply', 'adds'])
        >>> for v in t._values():
               print(v)

        add
        addition
        adds
        subtract
        multiply
        """
        node = node or self.head
        current = current or []
        if node.stop_state:
            # Here, reversal is more of a postprocessing step than a
            # preprocessing one: we're converting the reversed word stored in
            # the suffix tree back to its original order.
            yield self._maybe_reverse(self.postprocess(current))
        for key, node_ in node.edges.items():
            yield from self._values(current + [key], node_)

    def __iter__(self):
        """We separate this from self._values because we want the latter to
        be callable with arguments.
        """
        yield from self._values()

    def values(self):
        """Get a list of all sequences in the trie. User-facing version of
        `_values` that returns a list rather than a generator. User can also
        simply call `list(my_trie)` and get the same result.
        """
        return list(self)

    def __len__(self):
        # Don't just delegate to `self.values()` because __len__ is called
        # under the hood by list(self), thereby creating a recursion error in
        # `self.values()`. Could do `sum(1 for _ in self) but that gets slow
        # with large tries. There's currently no way to delete items so we
        # don't have to worry about length changing outside of `append`, and
        # if we do implement that we can simply adjust _length accordingly.
        return self._length

    def _startswith(self, seq, node=None):
        """Base behavior for both `startswith` and `endswith`.
        """
        # Validation occurs in `_find`.
        node = self._find(seq, node=node)
        if self.suffix:
            return [x + seq for x in self._values(node=node)]
        else:
            return [seq + x for x in self._values(node=node)]

    def startswith(self, seq, node=None):
        """Gets all values in a trie that start with a given sequence. (Unlike
        str.startswith, this does NOT return a boolean - consider renaming in
        a future version.)

        Parameters
        ----------
        seq: Iterable
            Same type as all the other sequences in the trie.
        node: TrieNode
            If provided, only the subtrie starting with this node will be
            searched. Defaults to the head, i.e. the whole trie will be
            searched.

        Returns
        -------
        list: Each item will be one of the sequences in the trie. If an empty
        list is returned, the trie contains no items sharing any leading
        values with the input `seq`.
        """
        if self.suffix:
            warnings.warn(
                'Suffix trees are optimized for the `endswith` method, but '
                '`startswith` will require walking the whole trie (may be '
                'slow). For an efficient implementation of `startswith`, you '
                'can create a prefix tree by passing `suffix=False` to '
                'Trie.__init__.'
            )
            if self.dtype == 'str':
                return [v for v in self._values(node=node)
                        if v.startswith(seq)]
            else:
                self._validate_input(seq)
                length = len(seq)
                return [v for v in self._values(node=node)
                        if v[:length] == seq]
        return self._startswith(seq, node=node)

    def endswith(self, seq, node=None):
        """Gets all values in a trie that end with a given sequence. (Unlike
        str.endswith, this does NOT return a boolean - consider renaming in
        a future version.)

        Parameters
        ----------
        seq: Iterable
            Same type as all the other sequences in the trie.
        node: TrieNode
            If provided, only the subtrie starting with this node will be
            searched. Defaults to the head, i.e. the whole trie will be
            searched.

        Returns
        -------
        list: Each item will be one of the sequences in the trie. If an empty
        list is returned, the trie contains no items sharing any trailing
        values with the input `seq`.
        """
        if not self.suffix:
            warnings.warn(
                'Prefix trees are optimized for the `startswith` method, but '
                '`endswith` will require walking the whole trie (may be '
                'slow). For an efficient implementation of `endswith`, you '
                'can create a suffix tree by passing `suffix=True` to '
                'Trie.__init__.'
            )
            if self.dtype == 'str':
                return [v for v in self._values(node=node)
                        if v.endswith(seq)]
            else:
                self._validate_input(seq)
                length = len(seq)
                return [v for v in self._values(node=node)
                        if v[-length:] == seq]
        return self._startswith(seq, node=node)

    def _longest_common_prefix(self, seq, seen):
        """Base functionality for the efficient version of
        `longest_common_prefix` for prefix trees and `longest_common_suffix`
        for suffix trees.

        Parameters
        ----------
        seq: Iterable
            Input sequence for which you wish to find sequences with matching
            prefixes (or suffixes). Type must match that of the other
            sequences in the trie.
        seen: list
            Empty list passed in. Seems to be necessary to accumulate matches.

        Returns
        -------
        list: Each item in the list is of the same tyep as `seq`. An empty
        list means no items in the tree share a common prefix with `seq`.
        """
        # Validation and reversal happens in `startswith`.
        matches = self.endswith(seq) if self.suffix else self.startswith(seq)
        if matches: return matches
        node = self.head
        for i, x in enumerate(self._maybe_reverse(seq)):
            if x in node.edges:
                seen.append(x)
                node = node.edges[x]
            elif i == 0:
                # Otherwise, all values are returned when the first item is
                # not in the trie.
                return []
            else:
                seen = self._maybe_reverse(self.postprocess(seen))
                if self.suffix:
                    matches = [v + seen for v in self._values(node=node)]
                else:
                    matches = [seen + v for v in self._values(node=node)]
                # Otherwise, we get bug where an empty list is returned if
                # the longest matching prefix is a complete sequence and the
                # node has no edges.
                if node.stop_state and not matches:
                    matches.append(seen)
                return matches

        # Case where the input sequence is present in the trie as a complete
        # sequence and it has no edges. This cannot be combined with the
        # case in the else statement above where matches is empty. We avoid
        # handling this upfront with something like
        # `if seq in self: return [seq]` because we want to capture additional
        # valid sequences in present.
        if node.stop_state:
            return [self._maybe_reverse(self.postprocess(seen))]

    def longest_common_prefix(self, seq):
        """Find sequences that share a common prefix with an input sequence.
        For instance, "carry" shares a common prefix of length 3 with "car",
        "carton", and "carsick", a common prefix of length 1 with "chat", and
        no common prefix with "dog". Note that a word shares a common prefix
        with itself, so if it's present in the trie it will be returned (in
        addition to any words that begin with that substring: for instance,
        both "carry" and "carrying" share a common prefix of length 5 with
        "carry".)

        Parameters
        ----------
        seq: Iterable
            Input sequence for which you wish to find sequences with matching
            prefixes. Type must match that of the other
            sequences in the trie.

        Returns
        -------
        list: Each item in the list is of the same tyep as `seq`. An empty
        list means no items in the tree share a common prefix with `seq`.
        """
        # Validation occurs in self.startswith, often via self._find.
        if not self.suffix:
            return self._longest_common_prefix(seq, [])

        warnings.warn(
            'Suffix trees are optimized for the `longest_common_suffix` '
            'method, but `longest_common_prefix` will require walking '
            'the whole trie (may be slow). For an efficient implementation '
            'of `longest_common_prefix`, you can create a prefix tree by '
            'passing `suffix=False` to Trie.__init__.'
        )
        self._validate_input(seq)
        res = []
        for i in range(len(seq), 0, -1):
            for v in self._values():
                if v[:i] == seq[:i]: res.append(v)
            if res: break
        return res

    def longest_common_suffix(self, seq):
        """Find sequences that share a common suffix with an input sequence.
        For instance, "carry" shares a common prefix of length 2 with "story",
        "tawdry", and "ornery", a common suffix of length 1 with "slowly", and
        no common suffix with "hate". Note that a word shares a common suffix
        with itself, so if it's present in the trie it will be returned (in
        addition to any words that end with that substring: for instance, both
        "carry" and "miscarry" share a common suffix of length 5 with
        "carry".)

        Parameters
        ----------
        seq: Iterable
            Input sequence for which you wish to find sequences with matching
            suffixes. Type must match that of the other sequences in the trie.

        Returns
        -------
        list: Each item in the list is of the same tyep as `seq`. An empty
        list means no items in the tree share a common prefix with `seq`.
        """
        # Validation and reversal occur in self.endswith, often via
        # self._find.
        if self.suffix:
            return self._longest_common_prefix(seq, [])

        warnings.warn(
            'Prefix trees are optimized for the `longest_common_prefix` '
            'method, but `longest_common_suffix` will require walking the '
            'whole trie (may be slow and memory intensive). For an '
            'efficient implementation of `longest_common_suffix`, you can '
            'create a suffix tree by passing `suffix=True` to Trie.__init__.'
        )
        self._validate_input(seq)
        res = []
        for i in range(len(seq), 0, -1):
            for v in self._values():
                if v[-i:] == seq[-i:]: res.append(v)
            if res: break
        return res

    def _validate_input(self, seq):
        """This should occur before calling self._maybe_reverse. Seq must be
        the same type as the other items in the trie or an error will be
        raised.
        """
        if type(seq).__name__ != self.dtype:
            raise TypeError('`seq` type doesn\'t match type of other '
                            'sequences.')
        if type(seq[0]).__name__ != self.child_dtype:
            raise TypeError('Type of first item in `seq` doesn\'t match type '
                            'of first item in other sequences.')

    def flip(self):
        """Flip trie from a prefix tree to a suffix tree or vice versa. This
        intentionally creates a new object rather than operating in place.

        Examples
        --------
        >>> pre_tree = Trie(['dog', 'cat', 'den', 'clean'], suffix=False)
        >>> suff_tree = pre_tree.flip()
        """
        return type(self)(self.values(), suffix=not self.suffix)

    def __repr__(self):
        # Display up to 5 values in repr.
        vals = self.values()
        if len(vals) > 5:
            vals = '[' + ', '.join(repr(v) for v in vals[:5]) + ', ...]'
        else:
            vals = str(vals)
        return f'{type(self).__name__}(values={vals}, suffix={self.suffix})'


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

    LSHDict does NOT support pickling as of version 6.0.6 (note: setitem seems
    to be called before init when unpickling, meaning we try to access
    self.forest in self._update_forest before it's been defined. Even if we
    change setitem so reindexing does not occur by default, it still tries to
    hash the new word and add it to the forest so unpickling will still fail).
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
            Number of similar candidates to retrieve. This uses Jaccard
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

    Pickling seems to work but I would use this with caution.

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

    def __dir__(self):
        return self.keys()


class PriorityQueue:
    """Creates list-like object that lets us retrieve the next item to process
    based on some priority measure, where larger priority values get processed
    first. This should be picklable.
    """

    def __init__(self, items=None):
        """
        Parameters
        ----------
        items: list[tuple[any, numbers.Real]]
            Each tuple must be structured as (item, priority) where a larger
            priority means that item will be processed sooner.
        """
        self._items = []
        if items:
            for item, priority in items:
                self.put(item, priority)

    def put(self, item, priority):
        heappush(self._items, (priority, item))

    def pop(self):
        try:
            return heappop(self._items)[-1]
        except IndexError:
            # The actual index error occurs due to the [-1] performed after
            # heappop, so we rewrite the error message to be more useful.
            raise IndexError('pop from empty queue')

    def __iter__(self):
        return self

    def __next__(self):
        """Makes PriorityQueue iterable, but note that this is a destructive
        action: our queue will be empty if we iterate over it
        (e.g. `for item in queue: pass`). Numeric indexing is not allowed
        since heapq only guarantees that the first item we retrieve will be
        the next one - it does not let us select the 3rd in line, for
        instance.
        """
        try:
            return self.pop()
        except IndexError:
            raise StopIteration

    def __contains__(self, key):
        return key in (item for priority, item in self._items)

    def __repr__(self):
        return f'{func_name(type(self))}({self._items})'


class IndexedDict(OrderedDict):
    """OrderedDict that lets us use integer indices. The tradeoff is that we
    can no longer use integers as keys since that would make it ambiguous
    whether we were trying to index in with a key or a positional index.

    This should be picklable.
    """

    def __init__(self, data=None):
        # Argument must be iterable.
        super().__init__(data or {})

    def __setitem__(self, key, val):
        if isinstance(key, int):
            raise TypeError('`key` must not be an integer.')
        super().__setitem__(key, val)

    def __getitem__(self, key):
        if isinstance(key, int):
            return list(self.values())[key]
        return super().__getitem__(key)


class LambdaDict(dict):
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
