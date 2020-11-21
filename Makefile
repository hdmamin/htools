.PHONY: dist pypi

dist:
	rm -rf dist/*
	python setup.py sdist

pypi: dist
	twine upload dist/*

