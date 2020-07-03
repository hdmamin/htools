.PHONY: dist pypi

dist:
	rm -rf dist/*
	python setup.py sdist

pypi:
	twine upload dist/*

