PKG_NAME = pytracker
CHECK_DIRS := $(PKG_NAME) setup.py
SOURCE_HOST := $(shell hostname)
TARGET_HOST = raspberry

all: check deploy

install: targetonly
	pip install -r requirements-prod.txt
	pip install -e .

installdev: sourceonly
	pip install -r requirements-dev.txt; \

check: sourceonly format imports lint test

format:
	black $(CHECK_DIRS)

imports:
	isort $(CHECK_DIRS)

lint:
	flake8 $(CHECK_DIRS)

test:
	python -m unittest discover -s $(PKG_NAME) -p "*_test.py"

run: install
	rm -rf images/*
	$(PKG_NAME) --loglevel debug --display :10.0

deploy: sourceonly
	rsync -avh --delete $(PKG_NAME) images Makefile requirements.txt requirements-prod.txt setup.py \
		pi@$(TARGET_HOST):~/Workspace/python/$(PKG_NAME)

sourceonly:
	@if [ "$(SOURCE_HOST)" != "fedora" ]; then \
		echo "Invalid environment: $(SOURCE_HOST)"; exit 1; \
	fi

targetonly:
	@if [ "$(SOURCE_HOST)" != "raspberrypi" ]; then \
		echo "Invalid environment: $(SOURCE_HOST)"; exit 1; \
	fi