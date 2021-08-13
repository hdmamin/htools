import os
from setuptools import setup, find_packages


def requirements(path='requirements.txt'):
    with open(path, 'r') as f:
        deps = [line.strip() for line in f]
    return deps


def version(path=os.path.join('htools', '__init__.py')):
    with open(path, 'r') as f:
        for row in f:
            if not row.startswith('__version__'):
                continue
            return row.split(' = ')[-1].strip('\n').strip("'")


setup(
    name='htools',
    version=version(),
    description='Harrison\'s custom functions.',
    packages=find_packages(include=['htools']),
    author='Harrison Mamin',
    zip_safe=False,
    install_requires=requirements(),
    extras_require={'fuzzy': ['fuzzywuzzy'],
                    'speedup': ['fuzzywuzzy[speedup]']},
    entry_points={'console_scripts': ['htools=htools.cli:cli']}
)

