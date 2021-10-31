PKG_NAME = tracker
NAMESPACE = ~/Workspace/piwatch
SOURCE_FILES = $(PKG_NAME) setup.py

SOURCE_HOST = fedora
TARGET_HOST = raspberry
EFFECTIVE_HOST = $(shell hostname)

LOG_LEVEL = debug
DISPLAY = :10.0

all: check deploy

sourceonly:
	@if [[ $(EFFECTIVE_HOST) != $(SOURCE_HOST) ]]; then \
		echo "Hostname mismatch: expected $(SOURCE_HOST), actual $(EFFECTIVE_HOST)"; \
		exit 1; \
	fi

targetonly:
	@if [[ $(EFFECTIVE_HOST) != $(TARGET_HOST) ]]; then \
		echo "Hostname mismatch: expected $(TARGET_HOST), actual $(EFFECTIVE_HOST)"; \
		exit 1; \
	fi

check: format imports lint test

format:
	black $(SOURCE_FILES)

imports:
	isort $(SOURCE_FILES)

lint:
	flake8 $(SOURCE_FILES)

.PHONY: test
test:
	python -m unittest discover -s $(PKG_NAME) -p "*_test.py"

installdev: sourceonly
	pip install -r requirements-dev.txt

install: targetonly
	pip install -r requirements-prod.txt
	pip install -e .

deploy: sourceonly
	rsync -avh --delete $(PKG_NAME) images Makefile requirements.txt requirements-prod.txt setup.py \
		pi@$(TARGET_HOST):$(NAMESPACE)/$(PKG_NAME)

run: targetonly
	rm -rf images/*
	$(PKG_NAME) --loglevel $(LOG_LEVEL) --display $(DISPLAY)
