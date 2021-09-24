PKG_NAME = pytracker
CHECK_DIRS = pytracker setup.py
SOURCE_HOST := $(shell hostname)
TARGET_HOST = raspberry3

all: check deploy

sourceonly:
	@if [ "$(SOURCE_HOST)" != "fedora" ]; then \
		echo "Invalid environment: $(SOURCE_HOST)"; exit 1; \
	fi

targetonly:
	@if [ "$(SOURCE_HOST)" != "raspberrypi" ]; then \
		echo "Invalid environment: $(SOURCE_HOST)"; exit 1; \
	fi

installdev: sourceonly
	pip install -r requirements-dev.txt; \

install: targetonly
	pip install -r requirements-prod.txt
	pip install -e .

check: format imports lint

format: sourceonly
	black $(CHECK_DIRS)

imports: sourceonly
	isort $(CHECK_DIRS)

lint: sourceonly
	flake8 $(CHECK_DIRS)

run: install
	rm -rf images/*
	$(PKG_NAME) --loglevel info --display :10.0

deploy: sourceonly
	rsync -avh --delete $(PKG_NAME) images Makefile requirements.txt requirements-prod.txt setup.py \
		pi@$(TARGET_HOST):~/Workspace/python/$(PKG_NAME)
