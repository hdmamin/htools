import os


CONFIG_DIR = os.path.expanduser(os.path.join('~', '.htools'))
CREDS_FILE = os.path.join(CONFIG_DIR, 'credentials.csv')
CONFIG_FILE = os.path.join(CONFIG_DIR, 'config.csv')


def get_default_user():
    """Get user's default email address. If one has not been set, user has the
    option to set it here.

    Returns
    --------
    str or None: A string containing the default email address. If user
        declines to specify one, None is returned.
    """
    os.makedirs(CONFIG_DIR, exist_ok=True)
    try:
        with open(CONFIG_FILE, 'r') as f:
            email = f.read().strip()
        return email

    except FileNotFoundError:
        cmd = input('No source email was specified and no default exists. '
                    'Would you like to add a default? [y/n]\n')
        if cmd == 'y':
            email = input('Enter default email address:\n')
            with open(CONFIG_FILE, 'w') as f:
                f.write(email)
            return email
        else:
            print('Exiting (no email specified).')
            return None


def get_credentials(from_email):
    """Get the user's password for a specified email address.

    Parameters
    ----------
    from_email: str
        The email address to get the password for.

    Returns
    -------
    str or None: If a password is found for the specified email address, return
        it as a string. Otherwise, return None.
    """
    # Load credentials.
    os.makedirs(CONFIG_DIR, exist_ok=True)
    try:
        with open(CREDS_FILE, 'r') as f:
            creds = dict([line.strip().split(',') for line in f])
            return creds[from_email]
    except Exception:
        cmd = input('We could not find credentials for that email '
                    'address. Would you like to enter your credentials '
                    'manually? [y/n]\n')

        # Case 1: User enters password manually.
        if cmd == 'y':
            password = input('Enter password:\n')
            cmd2 = input('Would you like to save these credentials locally '
                         '(if so, htools will remember your password next '
                         'time)? [y/n]\n')
            if cmd2 == 'y':
                with open(CREDS_FILE, 'a') as f:
                    f.write(f'{from_email},{password}\n')
                print(f'File saved to {CREDS_FILE}.')
            return password

    print('Exiting (no credentials given).')
    return None
