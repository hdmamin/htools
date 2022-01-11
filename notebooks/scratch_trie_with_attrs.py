"""
Want to eventually add support for trie with word-level and maybe char-level
attributes. Ex: word embeddings, word frequencies, char->char transition probs,
parts of speech, etc.). Also experimenting with a slightly different interface
than the existing trie in htools. Note that names are the same so if you import
htools * in a notebook, things may get confusing. Might want to rename these.
"""

from htools.core import listlike

class TrieNode:

    def __init__(self, edges=None, is_terminal=False, is_root=False, **kwargs):
        self.edges = edges or {}
        self.is_terminal = is_terminal
        self.is_root = is_root
        self.kwarg_names = set(kwargs)
        self.set_kwargs(**kwargs)

    def set_kwargs(self, **kwargs):
        self.kwarg_names.update(kwargs.keys())
        self.__dict__.update(**kwargs)

    def __contains__(self, char):
        return char in self.edges

    def __getitem__(self, char):
        return self.edges[char]

    def __setitem__(self, char, val):
        self.edges[char] = val

    def __repr__(self):
        res = f'TrieNode(edges={list(self.edges)}, '\
              f'is_terminal={self.is_terminal}, ' \
              f'is_root={self.is_root}'
        if self.kwarg_names:
            kwarg_str = ', '.join(f'{kwarg}={getattr(self, kwarg)}'
                                  for kwarg in self.kwarg_names)
            res += ', ' + kwarg_str
        return res + ')'


class Trie:

    def __init__(self, vocab=None):
        self.root = TrieNode(is_root=True)
        self._initialize(vocab)

    def _initialize(self, vocab):
        # Case 1: vocab is list/tuple. Must assign empty kwargs.
        if listlike(vocab):
            vocab = {word: {} for word in vocab}
        # Case 2: vocab is dict but values are not dicts. Must assign default name.
        elif not isinstance(next(iter(vocab.values())), dict):
            vocab = {word: {'val': val} for word, val in vocab.items()}
        for word, kwargs in vocab.items():
            self.add(word, **kwargs)

    def add(self, word, **kwargs):
        # These kwargs are associated with the whole word, e.g. if you want to
        # pass in word counts or word embeddings. Still need to implement support
        # for character-level attributes if I want that (e.g. if we want some kind of
        # transition probability from 1 character to the next).
        node = self.root
        for char in word:
            if char not in node:
                node[char] = TrieNode()
            node = node[char]
        node.is_terminal = True
        node.set_kwargs(**kwargs)

    def update(self, words):
        for word in words:
            self.add(word)


# TODO - eventually want method that yields nodes as we add/search for a new
# word. Based on my coroutine/generator pattern. Still debugging.
def _find(self, word):
   node = self.root
   yield
   for char in word:
       cur = yield node
       print('1', 'cur', cur, 'node', node)
       if cur:
           node = cur.get(char)
       print('2', 'cur', cur, 'node', node)


if __name__ == '__main__':
    word_dict = {
        'app': 18,
        'a': 6,
        'apple': 17,
        'about': 4,
        'able': 6,
        'zoo': 13,
        'zen': 11,
        'zesty': 14,
        'apply': 4,
        'cow': 18,
        'zigzag': 12
    }
    t = Trie(word_dict)
    coro = _find(t, 'app')
    print(next(coro))
    for x in coro:
        coro.send(x)
