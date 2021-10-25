NAMESPACE := ~/Workspace/piwatch
PKG_NAME = tracker
CHECK_DIRS := $(PKG_NAME) setup.py
SOURCE_HOST = fedora
TARGET_HOST = raspberry
EFFECTIVE_HOST := $(shell hostname)

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
		pi@$(TARGET_HOST):$(NAMESPACE)/$(PKG_NAME)

sourceonly:
	@if [ $(EFFECTIVE_HOST) != $(SOURCE_HOST) ]; then \
		echo "Invalid host environment: $(EFFECTIVE_HOST)"; exit 1; \
	fi

targetonly:
	@if [ $(EFFECTIVE_HOST) != $(TARGET_HOST) ]; then \
		echo "Invalid host environment: $(EFFECTIVE_HOST)"; exit 1; \
	fi
