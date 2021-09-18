PKG_NAME = pytracker
CHECK_DIRS = pytracker setup.py
HOST := $(shell hostname)

all: check deploy

sourceonly:
	@if [ "$(HOST)" != "fedora" ]; then \
		echo "Invalid environment: $(HOST)"; exit 1; \
	fi

targetonly:
	@if [ "$(HOST)" != "raspberrypi" ]; then \
		echo "Invalid environment: $(HOST)"; exit 1; \
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
	rm -rf screenshots/*
	$(PKG_NAME) --loglevel info --display :10.0

deploy: sourceonly
	rsync -avh --delete $(PKG_NAME) screenshots Makefile requirements.txt requirements-prod.txt setup.py \
		pi@raspberry2:~/Workspace/python/$(PKG_NAME)
