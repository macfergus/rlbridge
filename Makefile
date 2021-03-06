PYTHON ?= python

.PHONY: fix-imports
fix-imports:
	isort rlbridge/
	isort bridgecli

.PHONY: test
test:
	CUDA_VISIBLE_DEVICES=-1 $(PYTHON) -m unittest discover -p '*_test.py'

.PHONY: lint
lint:
	CUDA_VISIBLE_DEVICES=-1 pylint --rcfile=.pylintrc rlbridge/
