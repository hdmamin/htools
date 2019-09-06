from email.mime.text import MIMEText
from itertools import chain
import os
import pickle
import re
import smtplib
import sys
import time

from htools.config import GMAIL_ACCOUNT, GMAIL_CREDS_FILE


class LambdaDict(dict):
    """Create a default dict where the default function can accept parameters.
    Whereas the defaultdict in Collections can set the default as int or list,
    here we can pass in any function where the key is the parameter.
    """

    def __init__(self, default_function):
        super().__init__()
        self.f = default_function

    def __missing__(self, key):
        self[key] = self.f(key)
        return self[key]


def hdir(obj):
    """Print object methods and attributes, excluding magic methods."""
    return [x for x in dir(obj) if not x.startswith('_')]


def hmail(subject, message, to_email, from_email=GMAIL_ACCOUNT):
    """Send an email.

    :param from_email: Gmail address being used to send email.
    :param to_email: Recipient's email.
    :param subject: Subject line of email (str).
    :param message: Body of email (str).
    :return: None.
    """
    # Load credentials.
    with open(GMAIL_CREDS_FILE, 'r') as f:
        creds = dict([line.strip().split(',') for line in f])
    password = creds[from_email]

    # Create message instance.
    msg = MIMEText(message)
    msg['Subject'] = subject
    msg['From'] = from_email
    msg['To'] = to_email

    # Access server and send email.
    server = smtplib.SMTP(host='smtp.gmail.com', port=587)
    server.starttls()
    server.login(user=from_email, password=password)
    server.send_message(msg)
    print(f'Email sent to {to_email}.')


def htimer(func):
    """Provide conservative time estimate for a function to run."""
    def wrapper(*args, **kwargs):
        start = time.time()
        output = func(*args, **kwargs)
        print(f'\n[TIMER]: function <{func.__name__}> executed in roughly '
              f'{round(time.time() - start, 4)} seconds (conservatively).\n')
        return output
    return wrapper


def hsplit(text, sep, group=True, attach=True):
    """Flexible string splitting that retains the delimiter rather, unlike
    the built-in str.split() method.

    Parameters
    -----------
    text: str
        The input text to be split.
    sep: str
        The delimiter to be split on.
    group: bool
        Specifies whether to group consecutive delimiters together (True),
        or to separate them (False).
    attach: bool
        Specifies whether to attach the delimiter to the string that preceeds
        it (True), or to detach it so it appears in the output list as its own
        item (False).

    Returns
    --------
    list[str]

    Examples
    ---------
    text = "Score -- Giants win 6-5"
    sep = '-'

    # Case 0.1: Delimiters are grouped together and attached to the preceding
    word.
    >> hsplit(text, sep, group=True, attach=True)
    >> ['Score --', ' Giants win 6-', '5']

    # Case 0.2: Delimiters are grouped together but are detached from the
    preceding word, instead appearing as their own item in the output list.
    >> hsplit(text, sep, group=True, attach=False)
    >> ['Score ', '--', ' Giants win 6', '-', '5']

    Case 1.1: Delimiters are retained and attached to the preceding string.
    If the delimiter occurs multiple times consecutively, only the first
    occurrence is attached, and the rest appear as individual items in the
    output list.
    >> hsplit(text, sep, group=False, attach=True)
    >> ['Score -', '-', ' Giants win 6-', '5']

    # Case 1.2: Delimiters are retained but are detached from the preceding
    string. Each instance appears as its own item in the output list.
    >> hsplit(text, sep, group=False, attach=False)
    >> ['Score ', '-', '-', ' Giants win 6', '-', '5']
    """
    sep_re = re.escape(sep)
    regex = f'[^{sep_re}]*{sep_re}*'

    ##########################################################################
    # Case 0: Consecutive delimiters are grouped together.
    ##########################################################################
    if group:
        # Subcase 0.1
        if attach:
            return [word for word in re.findall(regex, text)][:-1]

        # Subcase 0.2
        else:
            return [word for word in re.split(f'({sep_re}+)', text) if word]

    ##########################################################################
    # Case 1: Consecutive delimiters are NOT grouped together.
    ##########################################################################
    words = text.split(sep)

    # Subcase 1.1
    if attach:
        return [word for word in re.findall(regex[:-1]+'?', text) if word]

    # Subcase 1.2
    return [word for word in chain(*zip(words, [sep]*len(words))) if word][:-1]


def print_object_sizes(space, limit=None, exclude_underscore=True):
    """Print the object names and sizes of the currently defined objects.

    Parameters
    -----------
    space: dict
        locals(), globals(), or vars()
    limit: int or None
        Optionally limit the number of objects displayed (default None for no
        limit).
    exclude_underscore: bool
        Determine whether to exclude objects whose names start with an
        underscore (default True).
    """
    var_size = [(var, sys.getsizeof(obj)) for var, obj in space.items()]
    for var, size in sorted(var_size, key=lambda x: -x[1])[:limit]:
        if not var.startswith('_') or not exclude_underscore:
            print(var, size)


def save_pickle(obj, fname, dir_name='data'):
    """Wrapper to quickly save a pickled object."""
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    path = os.path.join(dir_name, f'{fname}.pkl')
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    print(f'Data written to {path}.')


def load_pickle(fname, dir_name='data'):
    """Wrapper to quickly load a pickled object."""
    with open(os.path.join(dir_name, f'{fname}.pkl'), 'rb') as f:
        data = pickle.load(f)
    return data
