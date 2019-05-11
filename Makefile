PYTHON ?= python

.PHONY: fix-imports
fix-imports:
	isort -rc -y rlbridge/
	isort -rc -y bridgecli

.PHONY: test
test:
	$(PYTHON) -m unittest discover -p '*_test.py'

.PHONY: lint
lint:
	pylint --rcfile=.pylintrc rlbridge/
