from setuptools import setup, find_packages


with open('requirements.txt', 'r') as f:
    requirements = [line.strip() for line in f]

setup(name='htools',
      version='0.1',
      description='Harrison\'s custom functions.',
      packages=find_packages(include=['htools']),
      author='Harrison Mamin',
      zip_safe=False,
      install_requires=requirements)
