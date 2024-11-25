.PHONY: _prep create_environment requirements format lint docs docs-serve

## GLOBALS

PROJECT_NAME = cookiecutter-data-science
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python


###     UTILITIES
_prep:
	rm -f **/*/.DS_store


###     DEV COMMANDS

## Set up python interpreter environment
create_environment:
	conda create --name $(PROJECT_NAME) python=$(PYTHON_VERSION) -y
	@echo ">>> conda env created. Activate with:\nconda activate $(PROJECT_NAME)"

## Install Python Dependencies
requirements:
	$(PYTHON_INTERPRETER) -m pip install -r dev-requirements.txt

## Format the code using isort and black
format:
	isort --profile black ccds hooks tests docs/scripts
	black ccds hooks tests docs/scripts

lint:
	flake8 ccds hooks tests docs/scripts
	isort --check --profile black ccds hooks tests docs/scripts
	black --check ccds hooks tests docs/scripts


###     DOCS

docs:
	cd docs && mkdocs build

docs-serve:
	cd docs && mkdocs serve
