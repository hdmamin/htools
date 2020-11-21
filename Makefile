.PHONY: dist pypi

# version format: major:minor:patch
#bumpversion patch htools/__init__.py
bump:
	ifdef level
		bumpversion level htools/__init__.py
	else
		bumpversion patch htools/__init__.py
	endif

dist:
	rm -rf dist/*
	python setup.py sdist

pypi: dist
	twine upload dist/*

