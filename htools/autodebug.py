"""Importing this module will cause us to enter a debugging session whenever 
an error is thrown. It is not included in `from htools import *` because this
is rather extreme behavior.

Examples
--------
# Enable auto debugging session on error.
from htools import debug

# Revert to default python behavior.
autodebug.disable()
"""
import sys
from IPython import get_ipython
import pdb
import traceback

from htools.meta import monkeypatch


default_excepthook = sys.excepthook
ipy = get_ipython()


@monkeypatch(sys, 'excepthook')
def excepthook(type_, val, tb):
    """Replaces sys.excepthook when module is imported, which makes us enter
    a debugging session whenever an error is thrown. Disable by calling
    autodebug.disable().
    """
    traceback.print_exception(type_, val, tb)
    pdb.post_mortem(tb)


def ipy_excepthook(self, etype, evalue, tb, tb_offset):
    """IPython doesn't use sys.excepthook. We have to handle this separately
    and make sure it expects the right arguments.
    """
    return excepthook(etype, evalue, tb)


def disable():
    """Rever to default behavior.
    """
    sys.excepthook = default_excepthook
    # Tried doing `ipy.set_custom_exc((Exception,), None)` as suggested by
    # stackoverflow and chatgpt but it didn't quite restore the default
    # behavior. Manually remove this instead. I'm assuming only one custom
    # exception handler can be assigned for any one exception type and that
    # if we call disable(), we wish to remove the handler for Exception.
    ipy.custom_exceptions = tuple(x for x in ipy.custom_exceptions
                                  if x != Exception)


ipy.set_custom_exc((Exception,), ipy_excepthook)