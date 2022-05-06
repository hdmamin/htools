from abc import ABC, abstractmethod
import ast
from collections import ChainMap
from contextlib import contextmanager, redirect_stdout
from copy import copy, deepcopy
from functools import wraps, partial, update_wrapper
from fuzzywuzzy import fuzz, process
import importlib
import inspect
from inspect import Parameter, signature, _empty, getsource
import io
import json
import logging
import os
from pathlib import Path
import pkgutil
import signal
import ssl
import sys
from threading import Thread
import time
from tqdm.auto import tqdm
import types
import urllib
import warnings
from weakref import WeakSet

from htools.core import hdir, load, save, identity, hasstatic, tolist,\
    select, func_name
from htools.config import STD_LIB_GIST


class AutoInit:
    """Mixin class where child class has a long list of init arguments where
    the parameter name and the class attribute will be the same. Note that
    *args are not supported in the init method because each attribute that is
    defined in the resulting object must have a name. A variable length list
    of args can still be passed in as a single argument, of course, without the
    use of star unpacking.

    This updated version of AutoInit is slightly more user friendly than in V1
    (no more passing locals() to super()) but also slower and probably requires
    more testing (all because of the frame hack in the init method). Note that
    usage differs from the AutoInit present in htools<=2.0.0, so this is a
    breaking change.

    Examples
    --------
    Without AutoInit:

    class Child:
        def __init__(self, name, age, sex, hair, height, weight, grade, eyes):
            self.name = name
            self.age = age
            self.sex = sex
            self.hair = hair
            self.height = height
            self.weight = weight
            self.grade = grade
            self.eyes = eyes
        def __repr__(self):
            return f'Child(name={self.name}, age={self.age}, sex={self.sex}, '\
                   f'hair={self.hair}, weight={self.weight}, '\
                   f'grade={self.grade}, eyes={self.eyes})'

    With AutoInit:

    class Child(AutoInit):
        def __init__(self, name, age, sex, hair, height, weight, grade, eyes):
            super().__init__()

    Note that we could also use the following method, though this is less
    informative when constructing instances of the child class and does not
    have the built in __repr__ that comes with AutoInit:

    class Child:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    """

    def __init__(self):
        # Calculate how many frames to go back to get child class.
        frame_idx = type(self).__mro__.index(AutoInit)
        attrs = {k: v for k, v in sys._getframe(frame_idx).f_locals.items()
                 if not k.startswith('__')}
        attrs.pop('self')
        bound = signature(self.__class__.__init__).bind_partial(**attrs)

        # Flatten dict so kwargs are not listed as their own argument.
        bound.arguments.update(
            bound.arguments.pop('kwargs', {}).get('kwargs', {})
        )
        self._init_keys = set(bound.arguments.keys())
        for k, v in bound.arguments.items():
            try:
                setattr(self, k, v)
            except Exception as e:
                warnings.warn(f'Failed to set attribute {k}. {str(v)}')

    def __repr__(self):
        """Returns string representation of child class including variables
        used in init method. For the example in the class docstring, this would
        return:

        child = Child('Henry', 8, 'm', 'brown', 52, 70, 3, 'green')
        Child(name='Henry', age=8, sex='m', hair='brown', height=52,
              weight=70, grade=3, eyes='green')

        Returns
        -------
        str
        """
        fstrs = (f'{k}={repr(getattr(self, k))}' for k in self._init_keys)
        return f'{self.__class__.__name__}({", ".join(fstrs)})'


def auto_repr(cls):
    """Class decorator that provides __repr__ method automatically
    based on __init__ parameters. This aims to provide a simpler alternative
    to AutoInit that does not require access to the arguments passed to
    __init__. Attributes will only be included in the repr if they are in
    the class dict and appear in __init__ as a named parameter (with the
    same name).

    Examples
    --------
    @auto_repr
    class Foo:
        def __init__(self, a, b=6, c=None, p=0.5, **kwargs):
            self.a = a
            self.b = b
            # Different name to demonstrate that cat is not included in repr.
            self.cat = c
            # Property is not stored in class dict, not included in repr.
            self.p = p

        @property
        def p(self):
            return self._p

        @p.setter
        def p(self, val):
            if val > 0:
                self._p = val
            else:
                raise ValueError('p must be non-negative')

    >>> f = Foo(3, b='b', c='c')
    >>> f

    Foo(a=3, b='b')
    """

    def _repr(instance):
        args = dict(signature(instance.__init__).parameters)
        arg_strs = (f'{k}={repr(v)}' for k, v in instance.__dict__.items()
                    if k in args.keys())
        return f'{type(instance).__name__}({", ".join(arg_strs)})'

    cls.__repr__ = _repr
    return cls


def delegate(attr, iter_magics=False, skip=(), getattr_=True):
    """Decorator that automatically delegates attribute calls to an attribute
    of the class. This is a nice convenience to have when using composition.
    User can also choose to delegate magic methods related to iterables.

    Note: I suspect this could lead to some unexpected behavior so be careful
    using this in production.

    KNOWN ISSUES:
    -Max recursion error when a class inherits from nn.Module and
    delegates to the actual model.
    -Causes pickling issues at times. Haven't figured out cause yet.

    Parameters
    ----------
    attr: str
        Name of variable to delegate to.
    iter_magics: bool
        If True, delegate the standard magic methods related to iterables:
        '__getitem__', '__setitem__', '__delitem__', and '__len__'.
        # TODO: maybe consider adding __contains__? It most cases it should be
        fine - I believe python falls back to rely on __getitem__ - but if the
        object being delegated to defines some special __contains__ logic that
        is different than iterating using __getitem__, we might run into
        problems.
    skip: Iterable[str]
        Can optionally provide a list of iter_magics to skip. This only has
        an effect when `iter_magics` is True. For example, you may want to be
        able to iterate over the class but no allow item deletion. In this case
        you should pass skip=('__delitem__').
    getattr_: bool
        If True, delegate non-magic methods. This means that if you try to
        access an attribute or method that the object produced by the decorated
        class does not have, it will look for it in the delegated object.

    Examples
    --------
    Example 1: We can use BeautifulSoup methods like `find_all` directly on
    the Page object. Most IDEs should let us view quick documentation as well.

    @delegate('soup')
    class Page:
        def __init__(self, url, logfile, timeout):
            self.soup = self.fetch(url, timeout=timeout)
        ...

    page = Page('http://www.coursera.org')
    page.find_all('div')

    Example 2: Magic methods except for __delitem__ are delegated.

    @delegate('data', True, skip=('__delitem__'))
    class Foo:
        def __init__(self, data, city):
            self.data = data
            self.city = city

    >>> f = Foo(['a', 'b', 'c'], 'San Francisco')
    >>> len(f)
    3

    >>> for char in f:
    >>>     print(char)
    a
    b
    c

    >>> f.append(3); f.data
    ['a', 'b', 'c', 3]

    >>> del f[0]
    TypeError: 'Foo' object doesn't support item deletion

    >>> f.clear(); f.data
    []
    """
    def wrapper(cls):
        def _delegate(self, attr):
            """Helper that retrieves object that an instance delegates to.
            Just makes things a little easier to read here so we're not
            layering getattr calls too deeply.
            """
            return getattr(self, attr)

        # Any missing attribute will be delegated.
        if getattr_:
            def _getattr(self, new_attr):
                return getattr(_delegate(self, attr), new_attr)

            cls.__getattr__ = _getattr

        # If specified, delegate magic methods to make cls iterable.
        if iter_magics:
            if '__getitem__' not in skip:
                def _getitem(self, i):
                    return _delegate(self, attr)[i]

                setattr(cls, '__getitem__', _getitem)

            if '__contains__' not in skip:
                def _contains(self, i):
                    return i in _delegate(self, attr)

                setattr(cls, '__contains__', _contains)

            if '__setitem__' not in skip:
                def _setitem(self, i, val):
                    _delegate(self, attr)[i] = val

                setattr(cls, '__setitem__', _setitem)

            if '__delitem__' not in skip:
                def _delitem(self, i):
                    del _delegate(self, attr)[i]

                setattr(cls, '__delitem__', _delitem)

            if '__len__' not in skip:
                def _len(self):
                    return len(_delegate(self, attr))

                setattr(cls, '__len__', _len)
        return cls

    return wrapper


class LoggerMixin:
    """Mixin class that configures and returns a logger.

    Examples
    --------
    class Foo(LoggerMixin):

        def __init__(self, a, log_file):
            self.a = a
            self.log_file = log_file
            self.logger = self.get_logger(log_file)

        def walk(self, location):
            self.logger.info(f'walk received argument {location}')
            return f'walking to {location}'
    """

    def get_logger(self, path=None, fmode='a', level='info',
                   fmt='%(asctime)s [%(levelname)s]: %(message)s'):
        """
        Parameters
        ----------
        path: str or None
            If provided, this will be the path the logger writes to.
            If left as None, logging will only be to stdout.
        fmode: str
            Logging mode when using a log file. Default 'a' for
            'append'. 'w' will overwrite the previously logged messages. Note:
            this only affects what happens when we create a new logger ('w'
            will remove any existing text in the log file if it exists, while
            'a' won't. But calling `logger.info(my_msg)` twice in a row with
            the same logger will always result in two new lines, regardless of
            mode.
        level: str
            Minimum level necessary to log messages.
            One of ('debug', 'info', 'warning', 'error')
        fmt: str
            Format that will be used for logging messages. This uses the
            logging module's formatting language, not standard Python string
            formatting.

        Returns
        -------
        logging.logger
        """
        # When working in Jupyter, need to reset handlers.
        # Otherwise every time we run a cell creating an
        # instance of the logged class, the list of handlers will grow.
        logger = logging.getLogger(type(self).__name__)
        logger.handlers.clear()
        logger.setLevel(getattr(logging, level.upper()))

        # handler.basicConfig() doesn't work in Jupyter.
        formatter = logging.Formatter(fmt)
        handlers = [logging.StreamHandler(sys.stdout)]
        if path:
            # TODO: realized this breaks if we just pass in a file name,
            # e.g. tmp.log rather than logs/tmp.log.
            os.makedirs(os.path.dirname(path), exist_ok=True)
            handlers.append(logging.FileHandler(path, fmode))
        for handler in handlers:
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger


@delegate('logger')
class MultiLogger(LoggerMixin):
    """Easy way to get a pre-configured logger. This can also be used to
    record stdout, either through the context manager provided by contextlib
    or the function decorator defined in this module.

    It delegates to its logger and should be used as follows when explicitly
    called by the user:

    logger = MultiLogger('train.log')
    logger.info('Starting model training.'numeric)

    Notice we call the `info` method rather than `write`.
    """

    def __init__(self, path, fmode='w', fmt='%(message)s'):
        """
        Parameters
        ----------
        path: str or Path
            The log file to save to. If None is provided, will only log to
            stdout.
        fmode: str
            One of ('a', 'w'). See `LoggerMixin` docstring: this only affects
            behavior on the first write.
        fmt: str
            Message format. See `LoggerMixin` docstring for details.
        """
        self.logger = self.get_logger(path, fmode, 'info', fmt)

    def write(self, buf):
        """Provided for compatibility with `redirect_stdout` to allow logging
        of stdout while still printing it to the screen. The user should never
        call this directly.
        """
        if buf != '\n':
            self.logger.info(buf)


def verbose_log(path, fmode='w', fmt='%(message)s'):
    """Decorator to log stdout to a file while also printing it to the screen.
    Commonly used for model training.

    Parameters
    ----------
    path: str or Path
        Log file.
    fmode: str
        One of ('a', 'w') for 'append' mode or 'write' mode. Note that 'w' only
        overwrites the existing file once when the decorated function is
        defined: subsequent calls to the function will not overwrite previously
        logged content.
    fmt: str
        String format for logging messages. Uses formatting specific to
        `logging` module, not standard Python string formatting.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            fn_locals = bound_args(func, args, kwargs, True)
            logger = MultiLogger(path.format(**fn_locals), fmode, fmt)
            with redirect_stdout(logger):
                return func(*args, **kwargs)
        return wrapper
    return decorator


class SaveableMixin:
    """Provide object saving and loading methods. If you want to be able to
    pass a file name rather than a full path to `save`, the object can define
    a `self.dir` attribute.
    """

    def save(self, path=None, fname=None):
        """Pickle object with optional compression.

        Parameters
        ----------
        path: str or Path
            Path to save object to.
        fname: str or Path
            If passed in, method will use this as a filename within the
            object's `dir` attribute.
        """
        assert not fname or not path, 'Can\'t pass in both fname and path.'
        path = path or Path(self.dir) / fname
        save(self, path)

    @classmethod
    def load(cls, path):
        """Load object from pickle file.

        Parameters
        ----------
        path: str or Path
            Name of file where object is stored.
        """
        return load(path)


def chainmethod(func):
    """Decorator for methods in classes that want to implement
    eager chaining. Chainable methods should be instance methods
    that change 1 or more instance attributes and return None. All this
    decorator does is ensure these methods are called on a deep copy of the
    instance instead of on the instance itself so that operations don't affect
    the original object. The new object is returned.

    Examples
    --------
    @auto_repr
    class EagerChainable:

        def __init__(self, arr, b=3):
            self.arr = arr
            self.b = b

        @chainmethod
        def double(self):
            self.b *= 2

        @chainmethod
        def add(self, n):
            self.arr = [x+n for x in self.arr]

        @chainmethod
        def append(self, n):
            self.arr.append(n)

    >>> ec = EagerChainable([1, 3, 5, -22], b=17)
    >>> ec

    EagerChainable(arr=[1, 3, 5, -22], b=17)

    >>> ec2 = ec.append(99).double().add(400)
    >>> ec2

    EagerChainable(arr=[401, 403, 405, 378, 499], b=34)

    >>> ec   # Remains unchanged.
    EagerChainable(arr=[1, 3, 5, -22], b=17)
    """
    @wraps(func)
    def wrapper(instance, *args, **kwargs):
        new_inst = deepcopy(instance)
        func(new_inst, *args, **kwargs)
        return new_inst
    return wrapper


def lazychain(func):
    """Decorator to register a method as chainable within a
    LazyChainable class.
    """
    func._is_chainable = True
    @wraps(func)
    def wrapped(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapped


class LazyChainMeta(type):
    """Metaclass to create LazyChainable objects."""

    def __new__(cls, name, bases, methods):
        new_methods = {}

        # Find chainable staticmethods and create public versions.
        for k, v in methods.items():
            try:
                func = v.__get__(1)
                assert func._is_chainable
            except:
                continue
            public_name = k.lstrip('_')

            # Get args and kwargs passed to staticmethod (except for instance).
            sig = signature(func)
            sig = sig.replace(parameters=list(sig.parameters.values())[1:])

            # Must use default args so they are evaluated within loop.
            def make_public_method(func=func, private_name=k,
                                   public_name=public_name, sig=sig):
                def public(inst, *args, **kwargs):
                    bound = sig.bind(*args, **kwargs).arguments
                    new_method = partial(getattr(inst, private_name), **bound)
                    inst.ops.append(new_method)
                    return inst

                public.__name__ = public_name
                return public

            new_methods[public_name] = make_public_method()

        return type.__new__(cls, name, bases, {**methods, **new_methods})


class LazyChainable(metaclass=LazyChainMeta):
    """Base class that allows children to lazily chain methods,
    similar to a Spark RDD.

    Chainable methods must be decorated with @staticmethod
    and @chainmethod and be named with a leading underscore. A public
    method without the leading underscore will be created, so don't
    overwrite this with another method. Chainable methods
    accept an instance of the same class as the first argument,
    process the instance in some way, then return it. A chain of
    commands will be stored until the exec() method is called.
    It can operate either in place or not.

    Examples
    --------
    class Sequence(LazyChainable):

        def __init__(self, numbers, counter, new=True):
            super().__init__()
            self.numbers = numbers
            self.counter = counter
            self.new = new

        @staticmethod
        @lazychain
        def _sub(instance, n):
            instance.counter -= n
            return instance

        @staticmethod
        @lazychain
        def _gt(instance, n=0):
            instance.numbers = list(filter(lambda x: x > n, instance.numbers))
            return instance

        @staticmethod
        @lazychain
        def _call(instance):
            instance.new = False
            return instance

        def __repr__(self):
            pre, suf = super().__repr__().split('(')
            argstrs = (f'{k}={repr(v)}' for k, v in vars(self).items())
            return f'{pre}({", ".join(argstrs)}, {suf}'


    >>> seq = Sequence([3, -1, 5], 0)
    >>> output = seq.sub(n=3).gt(0).call().exec()
    >>> output

    Sequence(ops=[], numbers=[3, 5], counter=-3, new=False)

    >>> seq   # Unchanged because exec was not in place.

    Sequence(ops=[], numbers=[3, -1, 5], counter=0, new=True)


    >>> output = seq.sub(n=3).gt(-1).call().exec(inplace=True)
    >>> output   # None because exec was in place.
    >>> seq   # Changed

    Sequence(ops=[], numbers=[3, -1, 5], counter=-3, new=False)
    """

    def __init__(self):
        self.ops = []

    def exec(self, inplace=False):
        new = deepcopy(self)
        for func in self.ops:
            new = func(copy(new))
        # Clear ops list now that chain is complete.
        new.ops.clear()
        if inplace:
            self.__dict__ = new.__dict__
        else:
            self.ops.clear()
            return new

    def __repr__(self):
        argstrs = (f'{k}={repr(v)}' for k, v in vars(self).items())
        return f'{type(self).__name__}({", ".join(argstrs)})'


class ContextDecorator(ABC):
    """Abstract class that makes it easier to define classes that can serve
    either as decorators or context managers. This is a viable option if the
    function decorator case effectively wants to execute the function inside a
    context manager. If you want to do something more complex, this may not be
    appropriate since it's not clear what would happen in the context manager
    use case. Parentheses must be used in both cases (see examples).

    Examples
    --------
    import time

    class Timer(ContextDecorator):

        def __init__(self):
            # More complex decorators might need to store variables here.

        def __enter__(self):
            self.start = time.perf_counter()

        def __exit__(self, exc_type, exc_value, traceback):
            print('TIME:', time.perf_counter() - self.start)

    @Timer()
    def foo(a, *args):
        # do something

    with Timer():
        # do something

    # Both of these usage methods work!
    """

    def __call__(self, *args, **kwargs):
        """This method is NOT called when using child class as a context
        manager.
        """
        # Handle case where the decorated function is implicitly passed to the
        # decorator. Return the uncalled method just like how we often
        # `return wrapper` when writing a decorator as a function.
        if not hasattr(self, 'func'):
            self._wrap_func(args[0])
            return self.__call__

        self.__enter__()
        res = self.func(*args, **kwargs)
        self.__exit__(None, None, None)
        return res

    def _wrap_func(self, func):
        self.func = func
        update_wrapper(self, func)

    @abstractmethod
    def __enter__(self):
        """Do whatever you want to happen before executing the function (or
        the block of code inside the context manager).
        """

    @abstractmethod
    def __exit__(self, exc_type, exc_value, traceback):
        """Do anything that happens after the function finishes executing.
        The three arguments will all be None unless an error occurs.
        To suppress an error, this method must return True.
        """


class Stopwatch(ContextDecorator):
    """Print elapsed time in seconds during a function call or within a context
    manager. Because this is a ContextDecorator, you must explicitly
    instantiate this in both cases.

    @Stopwatch()
    def foo():
        # Do something.

    with Stopwatch():
        # Do something.

    """

    def __init__(self):
        self.thread = None
        self.running = False
        # Printing is not thread-safe so use logger instead. This format allows
        # us to update a single line rather than creating endless rows of
        # messages.
        self.logger = MultiLogger(None, fmt='\x1b[80D\x1b[1A\x1b[K%(message)s')

    def _start(self):
        """Update elapsed time every tenth of a second."""
        i = 1
        while self.running:
            time.sleep(.1)
            self.logger.info(f'Elapsed: {round(i * .1, 1)} sec')
            i += 1

    def start(self):
        """Start stopwatch in a new thread."""
        self.thread = Thread(target=self._start)
        self.thread.start()

    def stop(self):
        self.thread.join()

    def __enter__(self):
        self.running = True
        self.start()

    def __exit__(self, exc_type, exc_val, traceback):
        """Can't easily kill thread manually so we must tell it that we're no
        longer running.
        """
        self.running = False
        self.stop()


class class_or_instancemethod(classmethod):
    """Decorate a method so it can be called as both an instancemethod and a
    classmethod. The first argument to the method will be either the instance
    OR the class, depending on how it's called.

    Examples
    --------
    class Foo:

        @class_or_instancemethod
        def bar(self, x):
            if isinstance(self, type):
                # Classmethod functionality.
            else:
                # Instancemethod functionality.
    """

    def __get__(self, instance, cls):
        get = super().__get__ if instance is None else self.__func__.__get__
        return get(instance, cls)


class AbstractAttrs(type):
    """Basically the attribute equivalent of abc.abstractmethod: this allows
    us to define an abstract parent class that requires its children to
    possess certain class and/or instance attributes. This differs from
    abc.abstractproperty in a few ways:

    1. abstractproperty ignores instance attributes. AbstractAttrs lets us
    specify required instance attributes and/or class attributes and
    distinguish between the two.
    2. abstractproperty considers the requirement fulfilled by methods,
    properties, and class attributes. AbstractAttrs does not allow methods
    (including classmethods and staticmethods) to fulfill either requirement,
    though properties can fulfill either.

    Examples
    --------
    This class defines required instance attributes and class attributes,
    but you can also specify one or the other. If you don't care whether an
    attribute is at the class or instance level, you can simply use
    @abc.abstractproperty.

    class Parent(metaclass=AbstractAttrs,
                 inst_attrs=['name', 'metric', 'strategy'],
                 class_attrs=['order', 'is_val', 'strategy']):
        pass

    Below, we define a child class that fulfills some but not all requirements.

    class Child(Parent):
        order = 1
        metric = 'mse'

        def __init__(self, x):
            self.x = x

        @staticmethod
        def is_val(x):
            ...

        @property
        def strategy():
            ...

        def name(self):
            ...

    More specifically:

    Pass
    -possesses class attr 'order'
    -possess attribute 'strategy' (property counts as an instance attribute but
    not a class attribute. This is consistent with how it can be called:
    inst.my_property returns a value, cls.my_property returns a property
    object.)

    Fail
    -'metric' is a class attribute while our interface requires it to be a
    class attribute
    -'name' is a method but it must be an instance attribute
    -'is_val' is a staticmethod but it must be a class attribute
    """

    def __new__(cls, name, bases, methods, **meta_kwargs):
        """This provides user-defined parent classes with an
        `__init_subclass__` method that checks for class attributes. Errors
        will occur when the parent class is defined, not when instances of it
        are constructed.
        """
        class_ = type.__new__(cls, name, bases, methods)
        class_attrs = meta_kwargs.get('class_attrs', [])
        inst_attrs = meta_kwargs.get('inst_attrs', [])

        def __init_subclass__(cls, **kwargs):
            super().__init_subclass__(**kwargs)
            for attr in class_attrs:
                # TypeError maintains consistency with abstractmethod.
                # Remaining checks occur at instantiation.
                if not hasattr(cls, attr):
                    raise TypeError(f'{cls} must have class attribute '
                                    f'`{attr}`.')

        # Make sure we distinguish between the abstract parent class that
        # defines an interface and the child classes that implement it. The
        # abstract parent should not define the required attributes: it merely
        # enforces the requirement that its children do. We want the children
        # to inherit class_attrs and inst_attrs without overwriting them when
        # use them for validation. Only change this if you're very confident
        # you understand the repercussions.
        if class_attrs or inst_attrs:
            class_.__init_subclass__ = classmethod(__init_subclass__)
            class_._is_parent = True
            class_.class_attrs = class_attrs
            class_.inst_attrs = inst_attrs
        else:
            class_._is_parent = False
        return class_

    def __call__(cls, *args, **kwargs):
        """This is called when we create instances of our classes. Parents are
        initialized normally, while children undergo a series of checks for
        each of our required attributes.
        """
        inst = cls.__new__(cls, *args, **kwargs)
        if not isinstance(inst, cls): return inst

        inst.__init__(*args, **kwargs)
        if cls._is_parent: return inst

        # Validate children.
        for attr in inst.inst_attrs:
            # TypeError maintains consistency with abstractmethod.
            if not hasattr(inst, attr):
                raise TypeError(f'Instances of {type(inst)} must '
                                f'have instance attribute `{attr}`.')
            elif ismethod(getattr(inst, attr)):
                raise TypeError(f'`{attr}` must be an instance attribute, '
                                'not a method.')

        # In AbstractAttrs.__new__, methods are still unbound so we couldn't
        # easily check this until now.
        for attr in inst.class_attrs:
            # `ismethod` must check inst, not cls (cls.method is a function
            # while inst.method is a method). staticmethod can be retrieved
            # from either.
            if inspect.ismethod(getattr(inst, attr)) or hasstatic(inst, attr):
                raise TypeError(f'`{attr}` must be a class attribute, not a '
                                'method.')
            # property must be retrieved from cls, not inst.
            elif isinstance(getattr(cls, attr), property):
                raise TypeError(
                    f'`{attr}` must be a class attribute, not a property. '
                    'Properties fulfill instance attribute requirements but '
                    'not class attribute requirements.'
                )
        return inst


class Counted:
    """Add zero-index instance attribute "instance_num" tracking order in
    which instances were created. Class attribute "_instance_count" tracks the
    total number of instances of the class.

        class Bar(Counted):
    def __init__(self):
        super().__init__()
        self.x = x

    >>> b = Bar(3)
    >>> b2 = Bar(3)
    >>> b.instance_num, b2.instance_num
    0, 1
    >>> Bar._instance_count
    2
    """

    def __init_subclass__(cls, **kwargs):
        cls._instance_count = 0

    def __init__(self):
        self.instance_num = self._instance_count
        type(self)._instance_count += 1

    def __del__(self):
        self._instance_count -= 1


def counted(func):
    """Decorator to count the number of times a function has been called. The
    count updates AFTER the call completes.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        res = func(*args, **kwargs)
        wrapper.call_count += 1
        return res
    wrapper.call_count = 0
    return wrapper


def params(func):
    """Get parameters in a functions signature.

    Parameters
    ----------
    func: function

    Returns
    -------
    dict: Maps name (str) to Parameter.
    """
    return dict(signature(func).parameters)


def hasarg(func, arg):
    """Check if a function has a parameter with a given name. (Technically,
    hasparam might be a more appropriate name but hasarg lets us match the
    no-space convention of hasattr and getattr while maintaining readability.)

    Parameters
    ----------
    func: function
    arg: str
        The name of the parameter that you want to check for in func's
        signature.

    Returns
    -------
    bool: True if `func` has a parameter named `arg`.
    """
    return arg in params(func)


def bound_args(func, args, kwargs, collapse_kwargs=True):
    """Get the bound arguments for a function (with defaults applied). This is
    very commonly used when building decorators that log, check, or alter how
    a function was called.

    Parameters
    ----------
    func: function
    args: tuple
        Notice this is not *args. Just pass in the tuple.
    kwargs: dict
        Notice this is not **kwargs. just pass in the dict.
    collapse_kwargs: bool
        If True, collapse kwargs into the regular parameter dict. E.g.
        {'a': 1, 'b': True, 'kwargs': {'c': 'c_val', 'd': 0}} ->
        {'a': 1, 'b': True, 'c': 'c_val', 'd': 0}

    Returns
    -------
    OrderedDict[str, any]: Maps parameter name to passed value.
    """
    bound = signature(func).bind_partial(*args, **kwargs)
    bound.apply_defaults()
    args = bound.arguments
    if not collapse_kwargs: return args
    args.update(args.pop('kwargs', {}))
    return args


def handle_interrupt(func=None, cbs=(), verbose=True):
    """Decorator that allows us to interrupt a function with ctrl-c. We can
    pass in callbacks that execute on function end. Keep in mind that local
    variables will be lost as soon as `func` stops running. If `func` is a
    method, it may be appropriate to update instance variables while running,
    which we can access because the instance will be the first element of
    `args` (passed in as `self`).

    Notes:
    -Kwargs are passed to callbacks as a single dict, not as **kwargs.
    -A 'status_code' parameter tracks whether the last call was successful.
    The decorated function obviously can't reference the status of the current
    call since it's unknown until the function call completes, but the
    status is updated before executing any callbacks.

    Parameters
    ----------
    func: function
    cbs: Iterable[Callback]
        List of callbacks to execute when `func` completes. These will execute
        whether we interrupt or not.
    verbose: bool
        If True, print a message to stdout when an interrupt occurs.
    """
    if not func:
        return partial(handle_interrupt, cbs=tolist(cbs), verbose=verbose)
    func.status_code = 0
    for cb in cbs:
        cb.setup(func)
    @wraps(func)
    def wrapper(*args, **kwargs):
        func_inputs = bound_args(func, args, kwargs, collapse_kwargs=False)
        for cb in cbs:
            cb.on_begin(func, func_inputs)
        try:
            res = func(*args, **kwargs)
            wrapper.status_code = 0
        except KeyboardInterrupt:
            if verbose: print('KeyboardInterrupt. Aborting...')
            res = None
            wrapper.status_code = 1
        finally:
            for cb in cbs:
                cb.on_end(func, func_inputs, res)
        return res
    return wrapper


@contextmanager
def assert_raises(error):
    """Context manager to assert that an error is raised. This can be nice
    if we don't want to clutter up a notebook with error messages.

    Parameters
    ----------
    error: class inheriting from Exception or BaseException
        The type of error to catch, e.g. ValueError.

    Examples
    --------
    # First example does not throw an error.
    >>> with assert_raises(TypeError) as ar:
    >>>     a = 'b' + 6

    # Second example throws an error.
    >>> with assert_raises(ValueError) as ar:
    >>>     a = 'b' + 6

    AssertionError: Wrong error raised. Expected PermissionError, got
    TypeError(can only concatenate str (not "int") to str)

    # Third example throws an error because the code inside the context manager
    # completed successfully.
    >>> with assert_raises(ValueError) as ar:
    >>>     a = 'b' + '6'

    AssertionError: No error raised, expected PermissionError.
    """
    try:
        yield
    except error as e:
        print(f'As expected, got {error.__name__}({e}).')
    except Exception as e:
        raise AssertionError(f'Wrong error raised. Expected {error.__name__},'
                             f' got {type(e).__name__}({e}).') from None
    else:
        raise AssertionError(f'No error raised, expected {error.__name__}.')


class TimeExceededError(Exception):
    pass


def timebox_handler(time, frame):
    raise TimeExceededError('Time limit exceeded.')


@contextmanager
def timebox(seconds, strict=True, freq=.1, cleanup=True):
    """Try to execute code for specified amount of time before throwing error.
    If you don't want to throw an error, use with a try/except block.

    Parameters
    ----------
    seconds: float
        Max number of seconds before throwing error. This will be enforced with
        a relatively low level of precision.
    strict: bool
        If True, timeout will cause an error to be raised, halting execution of
        the entire program. If False, a warning message will be printed and
        the timeboxed operation will end, letting the program proceed to the
        next step.
    freq: float
        How often to update progress bar (measured in seconds).
    cleanup: bool
        If True, progress bar will disappear on function end. This is nice if
        we're calling the decorated function inside a loop and don't want
        hundreds of progress bars littering the notebook/terminal.

    Examples
    --------
    with time_box(5) as tb:
        x = computationally_expensive_code()

    More permissive version:
    x = step_1()
    with timebox(5) as tb:
        try:
            x = slow_step_2()
        except TimeExceededError:
            pass
    """

    def update_custom_pbar(signum, frame):
        """Handler that is called every `freq` seconds. User never calls this
        directly.
        """
        pbar.update(n=freq)
        if time.time() - pbar.start_t >= seconds:
            raise TimeExceededError('Time limit exceeded.')

    pbar = tqdm(total=seconds, bar_format='{l_bar}{bar}|{n:.2f}/{total:.1f}s',
                leave=not cleanup)
    try:
        signal.signal(signal.SIGALRM, update_custom_pbar)
        signal.setitimer(signal.ITIMER_REAL, freq, freq)
        yield
    except TimeExceededError as e:
        if strict: raise
        warnings.warn(e.args[0])
    finally:
        pbar.close()
        signal.alarm(0)


def timeboxed(time, strict=True, freq=.1):
    """Decorator version of timebox. Try to execute decorated function for
    `time` seconds before throwing exception.

    Parameters
    ----------
    time: float
        Max number of seconds before throwing error. This will be enforced with
        a relatively low level of precision.
    strict: bool
        If True, timeout will cause an error to be raised, halting execution of
        the entire program. If False, a warning message will be printed and
        the timeboxed operation will end, letting the program proceed to the
        next step.
    freq: float
        How often to update the progress bar (measured in seconds).

    Examples
    --------
    @timeboxed(5)
    def func(x, y):
        # If function does not complete within 5 seconds, will throw error.
    """
    def intermediate_wrapper(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with timebox(time, strict, freq) as tb:
                return func(*args, **kwargs)
        return wrapper
    return intermediate_wrapper


class cached_property:
    """Decorator for computationally expensive methods that should only be
    computed once (i.e. they take zero arguments aside from self and are slow
    to execute). Lowercase name is used for consistency with more decorators.
    Heavily influenced by example in `Python Cookbook` by David Beazley and
    Brian K. Jones. Note that, as with the @property decorator, no parentheses
    are used when calling the decorated method.

    Examples
    --------
    class Vocab:

        def __init__(self, tokens):
            self.tokens = tokens

        @cached_property
        def embedding_matrix(self):
            print('Building matrix...')
            # Slow computation to build and return a matrix of word embeddings.
            return matrix

    # First call is slow.
    >>> v = Vocab(tokens)
    >>> v.embedding_matrix

    Building matrix...
    [[.03, .5, .22, .01],
     [.4, .13, .06, .55]
     [.77, .14, .05, .9]]

    # Second call accesses attribute without re-computing
    # (notice no "Building matrix" message).
    >>> v.embedding_matrix

    [[.03, .5, .22, .01],
     [.4, .13, .06, .55]
     [.77, .14, .05, .9]]
    """

    def __init__(self, func):
        self.func = func

    def __get__(self, instance, cls):
        """This method is called when the variable being accessed is not in the
        instance's state dict. The next time the attribute is accessed, the
        computed value will be in the state dict so this method (and the method
        in the instance itself) is not called again unless the attribute is
        deleted.
        """
        # When attribute accessed as class method, instance is None.
        if instance is None:
            return self

        # When accessed as instance method, call method on instance as usual.
        # Then set instance attribute and return value.
        val = self.func(instance)
        setattr(instance, self.func.__name__, val)
        return val


class ReadOnly:
    """Descriptor to make an attribute read-only. This means that once a value
    has been set, the user cannot change or delete it. Note that read-only
    attributes must first be created as class variables (see example below).
    To allow more flexibility, we do allow the user to manually manipulate the
    instance dictionary.

    Examples
    --------
    class Dog:
        breed = ReadOnly()
        def __init__(self, breed, age):
            # Once breed is set in the line below, it cannot be changed.
            self.breed = breed
            self.age = age

    >>> d = Dog('dalmatian', 'Arnold')
    >>> d.breed

    'dalmatian'

    >>> d.breed = 'labrador'

    PermissionError: Attribute is read-only.

    >>> del d.breed

    PermissionError: Attribute is read-only.
    """

    def __init__(self):
        self.initialized = WeakSet()

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, cls):
        if instance is None:
            return self
        elif instance not in self.initialized:
            warnings.warn(
                f'Read-only attribute {self.name} has not been initialized.'
            )
        else:
            return instance.__dict__[self.name]

    def __set__(self, instance, value):
        if instance not in self.initialized:
            instance.__dict__[self.name] = value
            self.initialized.add(instance)
        else:
            raise PermissionError('Attribute is read-only.')

    def __delete__(self, instance):
        raise PermissionError('Attribute is read-only.')


def validating_property(func, allow_del=False):
    """Factory that makes properties that perform some user-specified
    validation when setting values. The returned function must be used as a
    descriptor to create a class variable before setting the instance
    attribute.

    Parameters
    ----------
    func: function
        Function or lambda that accepts a single parameter. This will be used
        when attempting to set a value for the managed attribute. It should
        return True if the value is acceptable, False otherwise.
    allow_del: bool
        If True, allow the attribute to be deleted.

    Returns
    -------
    function: A property with validation when setting values. Note that this
        will be used as a descriptor, so it must create a class variable as
        shown below. In the example, also notice that the name passed to
        LengthyInt mustt match the name of the variable it is assigned to.

    Examples
    --------
    LengthyInt = validating_property(
        lambda x: isinstance(x, int) and len(str(int)) > 4
    )

    class Foo:
        long = LengthyInt('long')
        def __init__(self, a, long):
            self.a = a
            self.long = long

    >>> foo = Foo(3, 4)

    ValueError: Invalid value 4 for argument long.

    # No error on instantiation because the argument is a valid LengthyInt.
    >>> foo = Foo(3, 543210)
    >>> foo.long

    543210

    >>> foo = Foo(3, 'abc')
    ValueError: Invalid value 'abc' for argument long.
    """
    def prop(name):
        @property
        def method(instance):
            return instance.__dict__[name]

        @method.setter
        def method(instance, val):
            if func(val):
                instance.__dict__[name] = val
            else:
                raise ValueError(f'Invalid value {val} for argument {name}.')

        if allow_del:
            @method.deleter
            def method(instance):
                del instance.__dict__[name]
        return method
    return prop


class Callback(ABC):
    """Abstract base class for callback objects to be passed to @callbacks
    decorator. Children must implement on_begin and on_end methods. Both should
    accept the decorated function's inputs and output as arguments

    Often, we may want to use the @debug decorator on one or both of these
    methods. If both methods should perform the same steps, one shortcut
    is to implement a single undecorated __call__ method, then have the
    debug-decorated on_begin and on_end methods return self(inputs, output).
    """

    @abstractmethod
    def setup(self, func):
        """
        Parameters
        ----------
        func: function
            The function being decorated.

        """

    @abstractmethod
    def on_begin(self, func, inputs, output=None):
        """
        Parameters
        ----------
        func: function
            The function being decorated.
        inputs: dict
            Dictionary of bound arguments passed to the function being
            decorated with @callbacks.
        output: any
            Callbacks to be executed after the function call can pass the
            function output to the callback. The default None value will remain
            for callbacks that execute before the function.
        """

    @abstractmethod
    def on_end(self, func, inputs, output=None):
        """
        Parameters
        ----------
        func: function
            The function being decorated.
        inputs: dict
            Dictionary of bound arguments passed to the function being
            decorated with @callbacks.
        output: any
            Callbacks to be executed after the function call can pass the
            function output to the callback. The default None value will remain
            for callbacks that execute before the function.
        """

    def __repr__(self):
        return f'{type(self).__name__}()'


def callbacks(cbs):
    """Decorator that attaches callbacks to a function. Callbacks should be
    defined as classes inheriting from abstract base class Callback that
    implement on_begin and on_end methods. This allows us to store states
    rather than just printing outputs or relying on global variables.

    Parameters
    ----------
    cbs: list
        List of callbacks to execute before and after the decorated function.

    Examples
    --------
    @callbacks([PrintHyperparameters(), PlotActivationHist(),
                ActivationMeans(), PrintOutput()])
    def train_one_epoch(**kwargs):
        # Train model.
    """
    def decorator(func):
        for cb in cbs:
            cb.setup(func)
        @wraps(func)
        def wrapper(*args, **kwargs):
            bound = signature(func).bind_partial(*args, **kwargs)
            bound.apply_defaults()
            for cb in cbs:
                cb.on_begin(func, bound.arguments, None)
            out = func(*args, **kwargs)
            for cb in cbs:
                cb.on_end(func, bound.arguments, out)
            return out
        return wrapper
    return decorator


def typecheck(func_=None, **types):
    """Decorator to enforce type checking for a function or method. There are
    two ways to call this: either explicitly passing argument types to the
    decorator, or letting it infer them using type annotations in the function
    that will be decorated. We allow multiple both usage methods since older
    versions of Python lack type annotations, and also because I feel the
    annotation syntax can hurt readability.

    Parameters
    ----------
    func_: function
        The function to decorate. When using decorator with
        manually-specified types, this is None. Underscore is used so that
        `func` can still be used as a valid keyword argument for the wrapped
        function.
    types: type
        Optional way to specify variable types. Use standard types rather than
        importing from the typing library, as subscripted generics are not
        supported (e.g. typing.List[str] will not work; typing.List will but at
        that point there is no benefit over the standard `list`).

    Examples
    --------
    In the first example, we specify types directly in the decorator. Notice
    that they can be single types or tuples of types. You can choose to
    specify types for all arguments or just a subset.

    @typecheck(x=float, y=(int, float), iters=int, verbose=bool)
    def process(x, y, z, iters=5, verbose=True):
        print(f'z = {z}')
        for i in range(iters):
            if verbose: print(f'Iteration {i}...')
            x *= y
        return x

    >>> process(3.1, 4.5, 0, 2.0)
    TypeError: iters must be <class 'int'>, not <class 'float'>.

    >>> process(3.1, 4, 'a', 1, False)
    z = a
    12.4

    Alternatively, you can let the decorator infer types using annotations
    in the function that is to be decorated. The example below behaves
    equivalently to the explicit example shown above. Note that annotations
    regarding the returned value are ignored.

    @typecheck
    def process(x:float, y:(int, float), z, iters:int=5, verbose:bool=True):
        print(f'z = {z}')
        for i in range(iters):
            if verbose: print(f'Iteration {i}...')
            x *= y
        return x

    >>> process(3.1, 4.5, 0, 2.0)
    TypeError: iters must be <class 'int'>, not <class 'float'>.

    >>> process(3.1, 4, 'a', 1, False)
    z = a
    12.4
    """
    # Case 1: Pass keyword args to decorator specifying types.
    if not func_:
        return partial(typecheck, **types)
    # Case 2: Infer types from annotations. Skip if Case 1 already occurred.
    elif not types:
        types = {k: v.annotation
                 for k, v in signature(func_).parameters.items()
                 if not v.annotation == inspect._empty}

    @wraps(func_)
    def wrapper(*args, **kwargs):
        fargs = signature(wrapper).bind(*args, **kwargs).arguments
        for k, v in types.items():
            if k in fargs and not isinstance(fargs[k], v):
                raise TypeError(
                    f'{k} must be {str(v)}, not {type(fargs[k])}.'
                )
        return func_(*args, **kwargs)
    return wrapper


def valuecheck(func):
    """Decorator that checks if user-specified arguments are acceptable.
    Because this re-purposes annotations to specify values rather than types,
    this can NOT be used together with the @typecheck decorator. Keep in mind
    that this tests for equality, so 4 and 4.0 are considered equivalent.

    Parameters
    ----------
    func: function
        The function to decorate. Use annotations to specify acceptable values
        as tuples, as shown below.

    Examples
    --------
    @valuecheck
    def foo(a, b:('min', 'max'), c=6, d:(True, False)=True):
        return d, c, b, a

    >>> foo(3, 'min')
    (True, 6, 'min', 3)

    >>> foo(True, 'max', d=None)
    ValueError: Invalid argument for parameter d. Value must be in
    (True, False).

    >>> foo('a', 'mean')
    ValueError: Invalid argument for parameter b. Value must be in
    ('min', 'max').
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        sig = signature(func)
        annos = {k: v.annotation for k, v in sig.parameters.items()}
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        for k, v in bound.arguments.items():
            choices = annos[k]
            if choices == inspect._empty: continue
            if v not in choices:
                raise ValueError(f'Invalid argument for parameter {k}. '
                                 f'Value must be in {choices}.')
        return func(*args, **kwargs)

    return wrapper


def deprecated(func=None, *, msg=''):
    """Decorator to mark a function as deprecated. This serves as both
    documentation (seeing it in the code is a good reminder) and also provides
    a warning if/when the function is called.

    Parameters
    ----------
    func: FunctionType
        Passed in to the decorator automatically.
    msg: str (optional)
        You may specify a more specific warning message to display when the
        function is called. This MUST be passed in as a keyword argument,
        not positional. If you don't mind using the default, use the
        no-parentheses form of the decorator.

    Examples
    --------
    @deprecated
    def my_old_func():
        # ...

    @deprecated(msg='My custom message!')
    def my_old_func():
        # ...
    """
    if func:
        assert callable(func), \
            '`deprecated` received a non-callable argument instead of a '\
            'function. If you meant to pass in msg, it must be a keyword arg.'
    else:
        return partial(deprecated, msg=msg)

    @wraps(func)
    def wrapper(*args, **kwargs):
        warnings.warn(
            msg or f'Soft deprecation: {func_name(func)} should not be used '
                   f'anymore.')
        return func(*args, **kwargs)
    return wrapper


def debug(func=None, prefix='', arguments=True, out_path=None):
    """Decorator that prints information about a function call. Often, this
    will only be used temporarily when debugging. Note that a wrapped function
    that accepts *args will display a signature including an 'args' parameter
    even though it isn't a named parameter, because the goal here is to
    explicitly show which values are being passed to which parameters. This
    does mean that the printed string won't be executable code in this case,
    but that shouldn't be necessary anyway since it would contain the same call
    that just occurred.

    The decorator can be used with or without arguments.

    Parameters
    ----------
    func: function
        Function being decorated.
    prefix: str
        A short string to prepend the printed message with. Ex: '>>>'
    arguments: bool
        If True, the printed message will include the function arguments.
        If False, it will print the function name but not its arguments.
    out_path: str or Path or None
        If provided, a dict of arguments will be saved as a json file as
        specified by this path. Intermediate directories will be created if
        necessary. Function arguments will be made available for string
        formatting if you wish to use that in the file name.
        Example: 'data/models/{prefix}/args.json'. The argument "prefix" will
        be used to save the file in the appropriate place. Note: `arguments`
        does not affect this since arguments are the only thing saved here.

    Examples
    --------
    Occasionally, you might pass arguments to different parameters than you
    intended. Throwing a debug_call decorator on the function helps you check
    that the arguments are matching up as expected. For example, the parameter
    names in the function below have an unexpected order, so you could easily
    make the following call and expect to get 8. The debug decorator helps
    catch that the third argument is being passed in as the x parameter.

    @debug
    def f(a, b, x=0, y=None, z=4, c=2):
        return a + b + c

    >>> f(3, 4, 1)
    CALLING f(a=3, b=4, x=1, y=None, z=4, c=2)
    9

    @debug(prefix='***', arguments=False)
    def f(a, b, x=0, y=None, z=4, c=2):
        return a + b + c

    >>> f(3, 4, 1)
    *** CALLING f()
    9
    """
    if not func:
        if prefix: prefix += ' '
        if out_path:
            assert str(out_path).endswith('.json'), \
                'out_path must ends with .json'
        return partial(debug, prefix=prefix, arguments=arguments,
                       out_path=out_path)

    @wraps(func)
    def wrapper(*args, **kwargs):
        out_fmt = '\n{}CALLING {}({})'
        arg_strs = ''
        if arguments:
            sig = bound_args(wrapper, args, kwargs, collapse_kwargs=True)
            if sig:
                first_key = next(iter(sig))
                # Remove self/cls arg from methods. Just check first arg to be
                # extra careful.
                if first_key in ('self', 'cls'):
                    del sig[first_key]
            arg_strs = (f'{k}={repr(v)}' for k, v in sig.items())

        # Print call message and return output.
        print(out_fmt.format(prefix, func.__qualname__, ', '.join(arg_strs)))
        if out_path: save(dict(sig), str(out_path).format(**sig))
        return func(*args, **kwargs)

    return wrapper


def log_stdout(func=None, fname=''):
    """Decorator that logs all stdout produced by a function.

    Parameters
    ----------
    func: function
        If the decorator is used without parenthesis, the function will be
        passed in as the first argument. You never need to explicitly specify
        a function.
    fname: str
        Path to log file which will be created. If None is specified, the
        default is to write to ./logs/wrapped_func_name.log. If specified,
        this must be a keyword argument.

    Examples
    --------
    @log_stdout
    def foo(a, b=3):
        print(a)
        a *= b
        print(a)
        return a**b

    @log_stdout(fname='../data/mylog.log')
    def foo(a, b=3):
        ...
    """
    if not func:
        return partial(log_stdout, fname=Path(fname))
    if not fname:
        fname = Path(f'./logs/{func.__name__}.log')

    @wraps(func)
    def wrapper(*args, **kwargs):
        os.makedirs(fname.parent, exist_ok=True)
        with open(fname, 'w') as f:
            with redirect_stdout(f):
                out = func(*args, **kwargs)
        return out

    return wrapper


def return_stdout(func):
    """Decorator that returns printed output from the wrapped function. This
    may be useful if we define a function that only prints information and
    returns nothing, then later decide we want to access the printed output.
    Rather than re-writing everything, we can slap a @return_stdout decorator
    on top and leave it as is. This should not be used if the decorated
    function already returns something else since we will only return what is
    printed to stdout. For that use case, consider the `log_stdout` function.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        res = io.StringIO()
        with redirect_stdout(res):
            func(*args, **kwargs)
        return res.getvalue()
    return wrapper


def log_cmd(path, mode='w', defaults=False):
    """Decorator that saves the calling command for a python script. This is
    often useful for CLIs that train ML models. It makes it easy to re-run
    the script at a later date with the same or similar arguments. If importing
    a wrapped function (or class with a wrapped method), you must include

    `os.environ['LOG_CMD'] = 'true'`

    in your script if you want logging to occur (accidentally overwriting log
    files unintentionally can be disastrous). Values 'True' and '1' also work
    but True and 1 do not (os.environ requires strings). Note that these values
    will not persist once the script completes.

    Parameters
    ----------
    path: str or Path
        Specifies file where output will be saved.
    mode: str
        Determines whether output should overwrite old file or be appended.
        One of ('a', 'w'). In most cases we will want append mode because we're
        tracking multiple trials.
    defaults: bool
        If True, include all arg values, even those that weren't specified
        from the command line (e.g. if your CLI function accepts up to 10 args
        (some with default values) and you pass in 3, the command will be
        logged as if you explicitly passed in all 10. This can be useful if
        you think your default args might change over time). If False, only
        args that were explicitly mentioned in your command will be used.

    Examples
    --------
    ```
    # train.py
    import fire

    @log_cmd('logs/training_runs.txt')
    def train(lr, epochs, dropout, arch, data_version, layer_dims):
        # Train model

    if __name__ == '__main__':
        fire.Fire(train)
    ```

    $ python train.py --lr 3e-3 --epochs 50 --dropout 0.5 --arch awd_lstm \
        --data_version 1 --layer_dims '[64, 128, 256]' \
        --dl_kwargs '{"shuffle": False, "drop_last": True}'

    After running the script with the above command, the file
    'logs/training_runs.txt' now contains a nicely formatted version of the
    calling command with a separate line for each argument name/value pair.

    We can also use variables that are passed to our function. All function
    args and kwargs will be passed to the string formatter so your variable
    names must match:

    @log_cmd('logs/train_run_v{version_number}.{ext}')
    def train(version_number, ext, epochs, arch='lstm'):
        # Train model
    """
    def decorator(func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            # Don't call when another function imports the wrapped function
            # unless we specifically ask it to by setting LOG_CMD = 'true' in
            # our CLI script. This is necessary because we sometimes want a
            # script to call another script and without this check, the new
            # command would always overwrite the old one which isn't always
            # what we want.
            if func.__module__ != '__main__' and not \
                        os.environ.get('LOG_CMD', '').lower() in ('true', '1'):
                return func(*args, **kwargs)

            # Log command before running script. Don't want to risk some
            # obscure bug occurring at the end and ruining a long process.
            fn_locals = bound_args(func, args, kwargs, True)
            start_of_line = ' \\\n\t'
            res = 'python'

            if defaults:
                res += __file__
                for k, v in fn_locals.items():
                    # Enclose data structure args in single quotes and use
                    # double quotes inside if necessary.
                    if isinstance(v, (tuple, list, dict, set)):
                        v = "'" + str(v).replace("'", '"') + "'"
                    res += f'{start_of_line}--{k} {v}'
            else:
                for arg in sys.argv:
                    res += start_of_line if arg.startswith('-') else ' '
                    # Ensure non-primitive kwargs are quoted appropriately.
                    for start, end in ['[]', '()', '{}']:
                        if arg.startswith(start) and arg.endswith(end):
                            arg = f"'{arg}'"
                    res += arg

            save(res+'\n\n', Path(path.format(**fn_locals)), mode)
            return func(*args, **kwargs)
        return wrapped
    return decorator


def wrapmethods(*decorators, methods=(), internals=False):
    """Class wrapper that applies 1 or more decorators to every non-magic
    method (properties are also excluded). For example, we often want @debug
    to be applied to many different methods.

    Parameters
    ----------
    decorators: callable
        1 or more decorators to apply to methods within a class. By default,
        methods with 1 or 2 leading underscores are excluded.
    methods: Iterable[str]
        Names of methods to wrap if you don't want to wrap all of them.
        Internal methods can be wrapped but magic methods and properties
        cannot.
    internals: bool
        If True, apply decorators to methods named with leading single
        underscores. This will be ignored if `methods` is specified.
    """

    def wrapper(cls):
        special_methods = (staticmethod, classmethod)
        if methods:
            to_wrap = dict.fromkeys(methods, True)
        else:
            to_wrap = {k: v == 'method' for k, v in
                       hdir(cls, False, internals=internals).items()}
        for attr, is_method in to_wrap.items():
            f = cls.__dict__[attr]
            if not is_method or isinstance(f, property):
                continue

            # Classmethod and staticmethod decorators need to be applied last.
            final_wrapper = identity
            if isinstance(f, special_methods):
                final_wrapper = type(f)
                f = f.__func__

            for d in decorators:
                f = d(f)
            setattr(cls, attr, final_wrapper(f))
        return cls
    return wrapper


def add_docstring(func):
    """Add the docstring from another function/class to the decorated
    function/class.

    Examples
    --------
    @add_docstring(nn.Conv2d)
    class ReflectionPaddedConv2d(nn.Module):
        ...
    """
    def decorator(new_func):
        new_func.__doc__ = f'{new_func.__doc__}\n\n{func.__doc__}'
        @wraps(new_func)
        def wrapper(*args, **kwargs):
            return new_func(*args, **kwargs)
        return wrapper
    return decorator


def timer(func):
    """Provide conservative time estimate for a function to run. Behavior may
    not be interpretable for recursive functions.

    Parameters
    -----------
    func: function
        The function to time.

    Examples
    ---------
    import time

    @timer
    def count_to(x):
        for i in range(x):
            time.sleep(0.5)

    >>> count_to(10)
    [TIMER]: count_to executed in approximately 5.0365 seconds.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        output = func(*args, **kwargs)
        duration = time.perf_counter() - start
        print(f'\n[TIMER]: {func.__name__} executed in approximately '
              f'{duration:.3f} seconds.\n')
        return output
    return wrapper


def handle(func=None, default=None):
    """Decorator that provides basic error handling. This is a rare decorator
    that is often most useful without the syntactic sugar: for instance,
    we may have a pre-existing function and want to apply it to a pandas Series
    while handling errors. See `Examples`.

    Parameters
    ----------
    func: callable
        The function to decorate.
    default: any
        This is the value that will be returned when the wrapped function
        throws an error.

    Examples
    --------
    There are a few different ways to use this function:

    @handle
    def func():
        # Do something

    @handle(default=0)
    def func():
        # Do something

    def some_func(x):
        # Do something
    df.name.apply(handle(some_func))
    """
    if not func: return partial(handle, default=default)
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception:
            return default
    return wrapper


@contextmanager
def block_timer(name=''):
    """Context manager to time a block of code. This works similarly to @timer
    but can be used on code outside of functions.

    Examples
    --------
    with block_timer() as bt:
        # Code inside the context manager will be timed.
        arr = [str(i) for i in range(25_000_000)]
        first = None
        while first != '100':
            arr.pop(0)
    print(bt['duration'])

    1.25   # Float measuring time spent in context manager.
    """
    data = {}
    if name: name = repr(name) + ' '
    start = time.perf_counter()
    try:
        yield data
    finally:
        duration = time.perf_counter() - start
        print(f'[TIMER]: Block {name}executed in {duration:.3f} seconds.')
        data['duration'] = duration


def count_calls(func):
    """Count the number of times a function has been called. The function can
    access this value inside itself through the attribute 'calls'. Note that
    counting is defined such that during the first call, func.calls already=1
    (i.e. it can be considered the n'th call, not that n calls have previously
    taken place not counting the current one).
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        calls = getattr(wrapper, 'calls', 0)
        wrapper.calls = calls + 1
        return func(*args, **kwargs)
    return wrapper


def min_wait(seconds):
    """Decorator that skips executing the decorated function if it was executed
    very recently (within a user-specified wait period).
    The resulting function's `last_called` attribute stores the value of
    perf_counter when it was last executed.

    Parameters
    ----------
    seconds: int or float
        Minimum wait period. If you try to execute the function < `wait`
        seconds after its last execution, it will return None.
    """
    if seconds >= 60 or seconds < 1:
        warnings.warn('min_wait is intended to be used with wait periods of a '
                      'few seconds. Wait periods under a second or over a '
                      'minute are not recommended - they may work but I '
                      'don\'t know.')
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_called = getattr(wrapper, 'last_called', float('-inf'))
            wrapper.last_called  = time.perf_counter()
            if wrapper.last_called - last_called < seconds:
                print(f'Not calling: function was called less than {seconds} '
                      'seconds ago.')
                return
            return func(*args, **kwargs)
        return wrapper
    return decorator


def copy_func(func):
    """Copy a function. Regular copy and deepcopy functionality do not work
    on functions the way they do on most objects. If we want to create a new
    function based on another without altering the old one (as in
    `rename_params`), this should be used.

    Parameters
    ----------
    func: function
        Function to duplicate.

    Returns
    -------
    function: Copy of input `func`.

    Examples
    --------
    def foo(a, b=3, *args, c=5, **kwargs):
        return a, b, c, args, kwargs

    foo2 = copy_func(foo)

    >>> foo2.__code__ == foo.__code__
    True

    >>> foo2 == foo
    False
    """
    new_func = types.FunctionType(func.__code__, func.__globals__,
                                  func.__name__, func.__defaults__,
                                  func.__closure__)
    defaults = getattr(func, '__kwdefaults__') or {}
    new_func.__kwdefaults__ = defaults.copy()
    return update_wrapper(new_func, func)


def rename_params(func, **old2new):
    """Rename one or more parameters. Docstrings and default arguments are
    updated accordingly. This is useful when working with code that uses
    `hasarg`. For example, my Incendio library uses parameter names
    to pass the correct arguments to different metrics.

    # TODO: looks like this updates the signature but doesn't actually change
    the variable names. So you can't call the decorated function with the
    new argument names.

    Parameters
    ----------
    func: function
        The old function to change.
    old2new: str
        One or more parameter names to change and their corresponding new
        names. See Example below for a more concrete example.

    Returns
    -------
    function: Same as input `func` but with updated parameter names.

    Examples
    --------
    def foo(a, b, *args, c=3, **kwargs):
        pass

    foo_metric = rename_params(func, a=y_true, b=y_pred)

    `foo_metric` will work exactly like `foo` but its first two parameters will
    now be named "y_true" and "y_pred", respectively. """
    new_func = copy_func(func)
    sig = signature(new_func)
    kw_defaults = func.__kwdefaults__ or {}
    names, params = map(list, zip(*sig.parameters.items()))
    for old, new in old2new.items():
        idx = names.index(old)
        default = kw_defaults.get(old) or params[idx].default
        params[idx] = inspect.Parameter(new, params[idx].kind, default=default)
    new_func.__signature__ = sig.replace(parameters=params)
    return new_func


def immutify_defaults(func):
    """Decorator to make a function's defaults arguments effectively immutable.
    We accomplish this by storing the initially provided defaults and assigning
    them back to the function's signature after each call. If you use a
    variable as a default argument, this does not mean that the variable's
    value will remain unchanged - it just ensures the initially provided value
    will be used for each call.
    """
    # If `__hash__` is not None, object is immutable already.
    # Python sets __defaults__ and __kwdefaults__ to None when they're empty.
    _defaults = tuple(o if getattr(o, '__hash__') else deepcopy(o)
                      for o in getattr(func, '__defaults__') or ()) or None
    _kwdefaults = {k: v if getattr(v, '__hash__') else deepcopy(v) for k, v
                   in (getattr(func, '__kwdefaults__') or {}).items()} or None

    @wraps(func)
    def wrapper(*args, **kwargs):
        res = func(*args, **kwargs)
        wrapper.__defaults__ = func.__defaults__ = deepcopy(_defaults)
        wrapper.__kwdefaults__ = func.__kwdefaults__ = deepcopy(_kwdefaults)
        return res
    return wrapper


@contextmanager
def temporary_globals(func, **kwargs):
    """Make a dict of key-value pairs temporarily available to a function in
    its global vars. We have to use function globals rather than globals()
    because the latter is evaluated when importing this function and so takes
    on the globals of htools/meta.py rather than of the scope where the
    code will ultimately be executed. Used in `add_kwargs` and `fallback`
    decorators (i.e. mostly for toy functionality, risky to actually use this).
    """
    old_globals = func.__globals__.copy()
    func.__globals__.update(kwargs)
    try:
        yield
    finally:
        for k in kwargs:
            if k in old_globals:
                func.__globals__[k] = old_globals[k]
            else:
                del func.__globals__[k]


def defined_functions(exclude=(), include_imported=False,
                      include_ipy_like=False):
    """Get all available functions defined in the current module.

    Parameters
    ----------
    exclude: Iterable[str]
        Names of any functions to exclude from results.
    include_imported: bool
        If True, include imported functions (this can be a LOT of functions if
        we've used star imports, as recommended for certain libraries like
        htools and fastai). If False, only return functions defined in the
        current module.
    include_ipy_like: bool
        Specifies whether to include functions whose names are like "_243"
        which is how IPython seems to store previously called functions
        (or something like that).

    Returns
    -------
    dict[str, FunctionType]: Dict mapping function name to function.
    """
    res = {}
    exclude = set(tolist(exclude))
    modules = vars(sys.modules['__main__']).copy()
    for k, v in modules.items():
        # IPython also sometimes has vars consisting only of underscores and
        # those become empty strings after the strip, which evaluate to not
        # numeric unless we add a digit.
        if isinstance(v, types.FunctionType) and \
                (include_imported or v.__module__ == '__main__') and \
                (include_ipy_like or not (k.strip('_')+'1').isnumeric()) and \
                k not in exclude:
            res[k] = v
    return res


def decorate_functions(decorator, exclude=(), include_imported=False,
                       include_ipy_like=False):
    """Decorate all (or some large subset, depending on args) functions
    available in the current module's global scope. Can be useful for
    debugging (see examples).

    Parameters
    ----------
    exclude: Iterable[str]
        Names of any functions to exclude from results.
    decorator: FunctionType
        The function that will be used to decorate all available functions.
        This must accept only a function as an argument, so make sure to pass
        in the appropriate object (for instance, if you want to use the
        htools.meta.timeboxed decorator, you must call it first with the
        desired arguments (e.g. time).
    include_imported: bool
        If True, include imported functions (this can be a LOT of functions if
        we've used star imports, as recommended for certain libraries like
        htools and fastai). If False, only return functions defined in the
        current module.
    include_ipy_like: bool
        Specifies whether to include functions whose names are like "_243"
        which is how IPython seems to store previously called functions
        (or something like that).

    Examples
    --------
    def foo(a):
        # Do something

    def bar(x, y):
        # Do something else

    if __name__ == '__main__':
        decorate_functions(debug)
        foo(3)
        bar(4, 5)
    """
    for k, v in defined_functions(exclude, include_imported,
                                  include_ipy_like).items():
        setattr(sys.modules['__main__'], k, decorator(v))


def register_functions(prefix):
    """Construct a dict of certain functions defined in a module. This lets
    scripts that import the module access these functions dynamically using a
    dict rather than using getattr messiness, importlib (which makes imports
    invisible to htools requirements.txt builder), or eval usage. See Examples.

    Parameters
    ----------
    prefix

    Returns
    -------
    dict[str, function]

    Examples
    --------
    # modeling.py

    def fit_knn(x, y, **kwargs):
        return knn(**kwargs).fit(x, y)

    def fit_nn(x, y, **kwargs):
        module = Network(**kwargs)
        module.train(x, y)
        return module

    def helper_function(z):
        return z

    FIT_FUNCS = register_functions(prefix='fit_')

    # train.py
    import fire
    from modeling import FIT_FUNCS

    def train(model):
        x, y = load_xy()
        FIT_FUNCS[model](x, y)

    if __name__ == '__main__':
        fire.Fire(train)
    """
    return {k.split(prefix)[-1]: v for k, v in defined_functions().items()
            if k.startswith(prefix)}


def source_code(name, lib_name='htools'):
    """Find the snippet of source code for a class/function defined in some
    library (usually htools). Like `inspect.getsource` except you just pass it
    strings and it handles all the imports.

    Warning: this was initially intended solely for use on htools-defined
    functionality, and it wasn't til afterwards that I realized it might extend
    reasonably well to other libraries. Known limitations: built in libraries
    (e.g. os) and big libraries with nested file structures (e.g. fastai)
    generally won't work.

    Parameters
    ----------
    name: str
        Class or function (usually defined in htools) that you want to see
        source code for.
    lib_name: str
        Name of library to check in, usually 'htools'.

    Returns
    -------
    tuple[str]: First item is the htools source code of the function/class
    (if not found, this is empty). Second item is a string that is either empty
    (if the function/class was found) or the name of a class/function most
    similar to the user-specified `name` if not.
    """
    if lib_name not in locals():
        lib = importlib.import_module(lib_name)
    else:
        lib = sys.modules[lib_name]

    # As of version 6.3.1, htools __init__ imports most modules so we can often
    # find the desired object as an attribute of the module itself. But we
    # might change that behavior in the future so the pkutil method is a good
    # fallback (and even now, it's needed for pd_tools methods).
    names = set()
    no_match = ''
    for mod, mod_name, _ in pkgutil.iter_modules(lib.__path__):
        try:
            module = getattr(lib, mod_name)
            src = getsource(getattr(module, name))
            return src, no_match
        except AttributeError as e:
            with open(lib.__path__[0] + f'/{mod_name}.py', 'r') as f:
                tree = ast.parse(f.read())
            names.update(x.name for x in tree.body
                         if isinstance(x, (ast.ClassDef, ast.FunctionDef)))
    backup = ''
    if names:
        backup = process.extract(name, names, limit=1, scorer=fuzz.ratio)[0][0]
    return no_match, backup


def fallback(meth=None, *, keep=(), drop=(), save=False):
    """Make instance/class attributes available as default arguments for a
    method. Kwargs can be passed in to override one or more of them. You can
    also choose for kwargs to update the instance attributes if desired.

    When using default values for keep/drop/save, the decorator can be used
    without parentheses. If you want to change one or more arguments, they
    must be passed in as keyword args (meth is never explicitly passed in, of
    course).

    Parameters
    ----------
    meth: method
        The method to decorate. Unlike the other arguments, this is passed in
        implicitly.
    keep: Iterable[str] or str
        Name(s) of instance attributes to include. If you specify a value
        here, ONLY these instance attributes will be made available as
        fallbacks. If you don't pass in any value, the default is for all
        instance attributes to be made available. You can specify `keep`,
        `drop`, or neither, but not both. This covers all possible options:
        keep only a few, keep all BUT a few, or keep all (drop all is the
        default case and doesn't require a decorator).
    drop: Iterable[str] or str
        Name(s) of instance attributes to ignore. I.e. if you want to make
        all instance attributes available as fallbacks except for self.df,
        you could specify drop=('df').
    save: bool
        If True, kwargs that share names with instance attributes will be
        overwritten with their new values. E.g. if we previously had
        self.lr = 3e-3 and you call your decorated method with
        obj.mymethod(lr=1), self.lr will be set to 1.

    Examples
    --------
    # Ex 1. self.a, self.b, and self.c are all available as defaults

    class Tree:
        def __init__(self, a, b, c=3):
            self.a = a
            self.b = b
            self.c = c

        @fallback
        def call(self, **kwargs):
            return a, b, c

    # Ex 2. self.b is not available as a default. We must put b in `call`'s
    # signature or the variable won't be accessible.

    class Tree:
        def __init__(self, a, b, c=3):
            self.a = a
            self.b = b
            self.c = c

        @fallback(drop=('b'))
        def call(self, b, **kwargs):
            return a, b, c

    # Ex 3. Self.b and self.c are available as defaults. If b or c are
    # specified in kwargs, the corresponding instance attribute will be updated
    # to take on the new value.

    class Tree:
        def __init__(self, a, b, c=3):
            self.a = a
            self.b = b
            self.c = c

        @fallback(keep=['b', 'c'], save=True)
        def call(self, a, **kwargs):
            return a, b, c
    """
    if meth is None:
        # Want to avoid errors if user passes in string or leaves comma out of
        # tuple when specifying keep/drop.
        return partial(fallback, keep=tolist(keep), drop=tolist(drop),
                       save=save)

    @wraps(meth)
    def wrapper(*args, **kwargs):
        self = args[0]
        self_kwargs = vars(self)
        if keep or drop: self_kwargs = select(self_kwargs, keep, drop)

        # Update kwargs with instance attribute defaults. Also update self if
        # user asked to save kwargs.
        for k, v in self_kwargs.items():
            if k not in kwargs:
                kwargs[k] = v
            elif save:
                setattr(self, k, kwargs[k])

        # Execute and return.
        with temporary_globals(meth, **kwargs):
            return meth(*args, **kwargs)

    return wrapper


def add_kwargs(*fns, required=True, variable=True):
    """When one or more functions are called inside another function, we often
    have the choice of accepting **kwargs in our outer function (downside:
    user can't see parameter names with quick documentation tools) or
    explicitly typing out each parameter name and default (downsides: time
    consuming and error prone since it's easy to update the inner function and
    forget to update the outer one). This lets us update the outer function's
    signature automatically based on the inner function(s)'s signature(s).
    The Examples section should make this more clear.

    The wrapped function must accept **kwargs, but you shouldn't refer to
    `kwargs` explicitly inside the function. Its variables will be made
    available essentially as global variables. This shares a related goal with
    fastai's `delegates` decorator but it provides a slightly different
    solution: `delegates` updates the quick documentation but the variables
    are still ultimately only available as kwargs. Here, they are available
    like regular variables.

    Note: don't actually use this for anything important, I imagine it could
    lead to some pretty nasty bugs. I was just determined to get something
    working.

    Parameters
    ----------
    fns: functions
        The inner functions whose signatures you wish to use to update the
        signature of the decorated outer function. When multiple functions
        contain a parameter with the same name, priority is determined by the
        order of `fns` (earlier means higher priority).
    required: bool
        If True, include required arguments from inner functions (that is,
        positional arguments or positional_or_keyword arguments with no
        default value). If False, exclude these (it may be preferable to
        explicitly include them in the wrapped function's signature).
    variable: bool
        If True, include *kwargs and **kwargs from the inner functions. They
        will be made available as {inner_function_name}_args and
        {inner_function_name}_kwargs, respectively (see Examples). Otherwise,
        they will be excluded.

    Examples
    --------
    def foo(x, c, *args, a=3, e=(11, 9), b=True, f=('a', 'b', 'c'), **kwargs):
        print('in foo')
        return x * c

    def baz(n, z='z', x='xbaz', c='cbaz'):
        print('in baz')
        return n + z + x + c

    baz comes before foo so its x param takes priority and has a default
    value of 'xbaz'. The decorated function always retains first priority so
    the c param remains positional despite its appearance as a positional
    arg in foo.

    @add_kwargs(baz, foo, positional=True)
    def bar(c, d=16, **kwargs):
        foo_res = foo(x, c, *foo_args, a=a, e=e, b=b, f=f, **foo_kwargs)
        baz_res = baz(n, z, x, c)
        return {'c': c, 'n': n, 'd': d, 'x': x, 'z': z, 'a': a,
                'e': e, 'b': b, 'f': f}

    bar ends up with the following signature:
    <Signature (c, n, d=16, x='xtri', foo_args=(), z='z', *, a=3, e=(11, 9),
                b=True, f=('a', 'b', 'c'), foo_kwargs={}, **kwargs)>

    Notice many variables are available inside the function even though they
    aren't explicitly hard-coded into our function definition. When using
    shift-tab in Jupyter or other quick doc tools, they will all be visible.
    You can see how passing in multiple functions can quickly get messy so
    if you insist on using this, try to keep it to 1-2 functions if possible.
    """
    param_types = {Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY}
    if required: param_types.add(Parameter.POSITIONAL_ONLY)

    def _args(fn):
        res = {}
        for k, v in params(fn).items():
            # If required=False, allow positional_or_keyword args with
            # defaults but not those without.
            if v.kind in param_types and (required
                                          or v.default != inspect._empty):
                res[k] = v

            # args/kwargs are converted to non-varying types and names are
            # adjusted to include function name. E.g. if we're adding kwargs
            # from function foo which accepts kwargs, that arg becomes a
            # keyword-only dictionary called foo_kwargs.
            elif variable:
                name = f'{fn.__name__}_{k}'
                if v.kind == Parameter.VAR_POSITIONAL:
                    kind = Parameter.POSITIONAL_OR_KEYWORD
                    default = ()
                elif v.kind == Parameter.VAR_KEYWORD:
                    kind = Parameter.KEYWORD_ONLY
                    default = {}
                else:
                    continue
                res[name] = Parameter(name, kind, default=default)
        return res

    # ChainMap operates in reverse order so functions that appear earlier in
    # `fns` take priority.
    extras_ = dict(ChainMap(*map(_args, fns)))

    def decorator(func):
        """First get params present in func's original signature, then get
        params from additional functions which are NOT present in original
        signature. Combine and sort param lists so positional args come first
        etc. Finally replace func's signature with our newly constructed one.
        """
        sig = signature(func)
        extras = [v for v in
                  select(extras_, drop=sig.parameters.keys()).values()]
        parameters = sorted(
            list(sig.parameters.values()) + extras,
            key=lambda x: (x.kind, x.default != inspect._empty)
        )
        func.__signature__ = sig.replace(parameters=parameters)

        @wraps(func)
        def wrapper(*args, **kwargs):
            """Execute wrapped function in a context where kwargs are
            temporarily available as globals. Globals will be restored to
            its prior state once execution completes.
            """
            # Order matters here: defaults must come first so user-passed
            # args/kwargs will override them.
            kwargs = {**{p.name: p.default for p in extras},
                      **func.__signature__.bind(*args, **kwargs).arguments}
            with temporary_globals(func, **kwargs):
                res = func(**kwargs)
            return res

        return wrapper

    return decorator


@valuecheck
def function_interface(present=(), required=(), defaults=(), startswith=(),
                       args: (True, False, None)=None,
                       kwargs: (True, False, None)=None,
                       like_func=None):
    """Decorator factory to enforce a some kind of function signature interface
    (i.e. the first two arguments must be ('model', 'x') or the function must
    accept **kwargs or the parameter 'learning_rate' must be present but not
    required because it has a default value).

    Parameters
    ----------
    present: Iterable[str]
        List of parameter names that must be present in the function signature.
        This will not check anything about their order or if they're required,
        just that they're present.
    required: Iterable[str]
        List of names that must be required parameters in the function (i.e.
        they have no default value).
    defaults: Iterable[str]
        List of names that must be present in the function signature with
        default values.
    startswith: Iterable[str]
        List of names that the function signature must start with. Order
        matters.
    args: bool
        If True, require function to accept *args. If False, require that it
        doesn't. If None, don't check either way.
    kwargs: bool
        If True, require function to accept **kwargs. If False, require that it
        doesn't. If None, don't check either way.
    like_func: None or function
        If provided, this function's signature will define the interface that
        all future decorated functions must match. Their name will obviously
        be different but all parameters must match (that means names, order,
        types, defaults, etc.).

    Returns
    -------

    """
    def decorator(func):
        def _param_status(param, params):
            if param not in params:
                return 'missing'
            if params[param].default == inspect._empty:
                return 'required'
            return 'optional'

        params = signature(func).parameters
        name = func_name(func)
        for param in present:
            if param not in params:
                raise RuntimeError(
                    f'`{name}` signature must include parameter {param}.'
                )
        for param in required:
            if _param_status(param, params) != 'required':
                raise RuntimeError(
                    f'`{name}` signature must include parameter {param} with '
                    'no default parameter.'
                )
        for param in defaults:
            if _param_status(param, params) != 'optional':
                raise RuntimeError(
                    f'`{name}` signature must include parameter {param} with '
                    'default value.'
                )
        params_list = list(params.keys())
        for i, param in enumerate(startswith):
            if params_list[i] != param:
                raise RuntimeError(f'`{name}` signature\'s parameter #{i+1} '
                                   f'(1-indexed) must be named {param}.')
        if args is not None:
            has_args = any(v.kind == Parameter.VAR_POSITIONAL
                           for v in params.values())
            if has_args != args:
                raise RuntimeError(f'`{name}` signature must '
                                   f'{"" if args else "not"} accept *args.')
        if kwargs is not None:
            has_kwargs = any(v.kind == Parameter.VAR_KEYWORD
                             for v in params.values())
            if has_kwargs != kwargs:
                raise RuntimeError(
                    f'`{name}` signature must {"" if kwargs else "not"} '
                    'accept **kwargs.'
                )
        if like_func and str(signature(like_func)) != str(signature(func)):
            raise RuntimeError(f'`{name}` signature must match {like_func} '
                               'signature.')

        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    return decorator


def Lazy(func=None, *, lazy=True):
    """"Decorator that provides a function with a boolean parameter "lazy".
    When set to true, the function will not be executed yet, sort of like a
    coroutine (see examples). This can be nice for testing purposes. Also
    opens the door to some interesting things (maybe sort of allows for
    decorated objects? Not sure of all applications yet). Would have preferred
    a lowercase function name but I also want to keep the parameter name as
    "lazy" while avoiding confusion.

    Examples
    --------
    @Lazy
    def foo(a, b=3):
        return a * b

    >>> foo(2, lazy=False)
    6

    >>> res = foo(2)
    >>> res()
    6

    In the second example, notice we didn't get any output until explicitly
    calling the result. Also note that we can change the default mode by using
    the decorator like (keyword argument, not positional):
    @Lazy(lazy=False)
    """
    if func is None: return partial(Lazy, lazy=lazy)
    if 'lazy' in params(func):
        raise RuntimeError(
            f'Decorated function {func} must not have parameter named "lazy".'
            'It will be inserted automatically.'
        )
    @wraps(func)
    def wrapper(*args, lazy=lazy, **kwargs):
        if lazy:
            return lambda: func(*args, **kwargs)
        return func(*args, **kwargs)
    return wrapper


def mark(**kwargs):
    """Decorator to mark a function or method with various attributes. For
    example, we might want to mark all methods of a class that are called
    internally by a particular method, or all the methods that are used for
    feature engineering, or all the methods that make http calls.

    Parameters
    ----------
    kwargs: (str, any)
        These will be used to set attributes of the function or method.

    Examples
    --------
    class FooBar:

        @mark(http=True, priority=1)
        def foo(self, x):
            ...

        @mark(priority=2)
        def bar(self, x):
            ...
    """
    def decorator(func):
        for k, v in kwargs.items():
            setattr(func, k, v)
        @wraps(func)
        def wrapper(*args, **kwargs_):
            return func(*args, **kwargs_)
        return wrapper
    return decorator


def coroutine(coro):
    """Decorator to prime a coroutine (lets us avoid calling coro.send(None)
    before sending actual values).
    """
    @wraps(coro)
    def wrapper(*args, **kwargs):
        # Note that this is only executed once when we first create the
        # coroutine - subsequent interactions use `send` on the existing
        # object.
        res = coro(*args, **kwargs)
        res.send(None)
        return res
    return wrapper


@counted
def in_standard_library(package_name):
    """Define this in `meta` module since we require the `counted` decorator
    so we only make the http request the first time we execute the function
    (had trouble getting packaging arg `data_files` in setup.py to work as
    expected). Useful for determining what pip packages need to be installed in
    a project (if a package isn't built in, we presumably need to install it).

    Note to self: we could also implement this like:

    @mark(library=StandardLibrary())
    def in_standard_library(package_name):
        return package_name in in_standard_library.library

    where StandardLibrary is a class with a cached_property-decorated method
    to fetch the library list and a __contains__ method that delegates checks
    to the resulting attribute produced by the descriptor. Still deciding which
    pattern I prefer for the "do something on the first call" use case.

    Parameters
    ----------
    package_name: str
        Name of a package, e.g. numpy.

    Returns
    -------
    bool: True if package is included in the standard library, False otherwise.
    """
    global STANDARD_LIBRARY
    if in_standard_library.call_count == 0:
        # Brew-installed versions of python sometimes don't include the
        # necessary certificate for http requests to work.
        ssl._create_default_https_context = ssl._create_unverified_context
        r = urllib.request.urlopen(STD_LIB_GIST)
        STANDARD_LIBRARY = json.loads(r.read())
    return package_name in STANDARD_LIBRARY


def get_module_docstring(path, default=''):
    """Got the module level docstring from a python file.

    Parameters
    ----------
    path: str or Path
        File to extract docstring from. You can also pass in __file__ to get
        the current file.
    default: str
        Backup value when module has no docstring.

    Returns
    -------
    str
    """
    with open(path, 'r') as f:
        tree = ast.parse(f.read())
    return ast.get_docstring(tree) or default


def getindex(arr, val, default=-1):
    """Like list.index but allows us to specify a fallback value if the value
    isn't present in the list, kind of like getattr.

    Parameters
    ----------
    arr: list
        The list to search in.
    val: any
        The item to search for.
    default: any
        Value to return if val is not in arr.
    """
    return arr.index(val) if val in arr else default


def set_module_global(module, key, value):
    """Create global variable in an imported module. This is a slightly hacky
    workaround that solves some types of circular imports.

    Parameters
    ----------
    module: str
        Name of module to create variable in.
    key: str
        Name of variable to create in module.
    value: any
        Value of variable to create in module.
    """
    module_ = sys.modules[module]
    if hasattr(module_, key):
        warnings.warn(f'{module} has existing variable {key} that will be '
                      f'overwritten.')
    setattr(module_, key, value)


def set_module_globals(module, **kwargs):
    """Set multiple global variables in an imported module.

    Parameters
    ----------
    module: str
        Module name.
    kwargs: any
        (Key, value) pairs.
    """
    for k, v in kwargs.items():
        set_module_global(module, k, v)


class Partial:
    """More powerful (though also potentially more fragile) version of
    functools.partial that updates the resulting signature to work better with
    Jupyter's quick documentation feature. We also update __repr__, __str__,
    and __name__ attributes (optionally renaming the source function). Unlike
    functools.partial, we also reorder parameters so that those without
    defaults always come before those with defaults.

    Note: the resulting object is actually a callable class, not a function.
    """

    def __init__(self, func, name=None, **kwargs):
        """
        Parameters
        ----------
        func: function
        name: str or None
            If None, the source function's name will be used.
        kwargs: any
            Default arguments to set, like in functools.partial.
        """
        self.func = copy_func(func)
        self.old_name = func.__name__

        # Track names of positional args in old function since this affects
        # the order args must be passed in if var_positional parameters
        # (*args) are present.
        self.old_pos_pars = []
        self.kwargs_name = ''
        self.args_name = ''
        new_pars = []
        old_sig = signature(self.func)
        for k, v in old_sig.parameters.items():
            # Check parameter kind for error handling and argument resolution
            # in __call__.
            if v.kind == 0:
                raise NotImplementedError(
                    'rigorous_partial does not support functions with '
                    'positional only parameters.'
                )
            elif v.kind == 2:
                self.args_name = k
            elif v.kind == 4:
                self.kwargs_name = k
                break

            if v.kind <= 2:
                self.old_pos_pars.append(k)

            # Assign default value from newly specified kwargs if provided.
            if k in kwargs:
                default = kwargs.pop(k)
                kind = 3
            else:
                default = v.default
                kind = v.kind
            param = Parameter(k, kind, default=default)
            new_pars.append(param)

        # Remaining kwargs only: those that were not present in func's
        # signature. Require that they be keyword only since ordering can
        # cause issues (updating signature affects what we see but doesn't
        # seem to affect the actual order args are passed in, presumably due
        # to old __code__ object).
        for k, v in kwargs.items():
            param = Parameter(k, 3, default=v)
            new_pars.append(param)
        if self.kwargs_name:
            new_pars.append(Parameter(self.kwargs_name, 4))

        # Ensure we don't accidentally place any parameters with defaults
        # ahead of those without them. Third item in tuple is a tiebreaker
        # (defaults to original function's parameter order).
        old_names = [p for p in old_sig.parameters]
        new_pars.sort(
            key=lambda x: (x.kind, x.default != _empty,
                           getindex(old_names, x.name, float('inf')))
        )

        # I honestly forget why we need to set the attribute on self.func too,
        # I just remember it was needed to resolve a bug (I think it was
        # related to *args resolution).
        self.__signature__ = self.func.__signature__ = old_sig.replace(
            parameters=new_pars
        )
        self.__defaults__ = tuple(p.default for p in new_pars if p.kind < 3
                                  and p.default != _empty)
        self.__kwdefaults__ = {p.name: p.default for p in new_pars
                               if p.kind == 3}
        if name: self.func.__name__ = name
        update_wrapper(self, self.func)

    def __call__(self, *args, **new_kwargs):
        # Remember self.func's actual code is unchanged: we updated how its
        # signature appears, but that doesn't affect the actual mechanics.
        # Therefore, we need to carefully resolve args and kwargs so that the
        # function is called so that behavior matches what we'd expect based
        # on the order shown in the signature.
        tmp_kwargs = bound_args(self.func, args,
                                {**self.__kwdefaults__, **new_kwargs})
        final_args = {name: tmp_kwargs.pop(name)
                      for name in self.old_pos_pars}
        final_star_args = final_args.pop(self.args_name, [])
        final_kwargs = select(tmp_kwargs, drop=list(final_args))
        return self.func(*final_args.values(), *final_star_args,
                         **final_kwargs)

    def __repr__(self):
        """Note: the memory address here points to that of the copy of the
        source function stored in self.func.
        """
        return repr(self.func).replace(self.old_name, self.__name__)

    def __str__(self):
        return str(self.func).replace(self.old_name, self.__name__)


class ReturningThread(Thread):
    """
    # TODO: consider creating a separate concurrency.py module? Have to
    consider dependencies, e.g. this uses meta.add_docstring, so meta.py would
    not be able to use functionality from concurrency.py.
    """

    @add_docstring(Thread)
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs=None, *, daemon=None):
        """This is identical to a regular thread except that the join method
        returns the value returned by your target function. The
        Thread.__init__ docstring is shown below for the sake of convenience.
        """
        super().__init__(group=group, target=target, name=name,
                         args=args, kwargs=kwargs, daemon=daemon)
        self.result = None

    def run(self):
        self.result = self._target(*self._args, **self._kwargs)

    def join(self, timeout=None):
        super().join(timeout)
        return self.result
