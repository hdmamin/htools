from email.mime.text import MIMEText
import smtplib
import sys
import time


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


def hmail(subject, message, to_email, from_email='hmamin55@gmail.com'):
    """Send an email.

    :param from_email: Gmail address being used to send email.
    :param to_email: Recipient's email.
    :param subject: Subject line of email (str).
    :param message: Body of email (str).
    :return: None.
    """
    # Load credentials.
    with open('/Users/hmamin/creds/gmail.csv', 'r') as f:
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
