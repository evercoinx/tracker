PKG_NAME = tracker
NAMESPACE = ~/Workspace/evercoinx
SOURCE_FILES = $(PKG_NAME) setup.py

SOURCE_HOST = fedora
TARGET_HOST = raspberry
EFFECTIVE_HOST = $(shell hostname)

DISPLAY = :10.0
SCREEN_TOP_MARGIN = 98

all: check deploy

sourceonly:
	@if [ "$(EFFECTIVE_HOST)" != "$(SOURCE_HOST)" ]; then \
		echo "Hostname mismatch: expected $(SOURCE_HOST), actual $(EFFECTIVE_HOST)"; \
		exit 1; \
	fi

targetonly:
	@if [ "$(EFFECTIVE_HOST)" != "$(TARGET_HOST)" ]; then \
		echo "Hostname mismatch: expected $(TARGET_HOST), actual $(EFFECTIVE_HOST)"; \
		exit 1; \
	fi

check: format imports lint

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
	rsync -avh $(PKG_NAME) stream Makefile requirements.txt requirements-prod.txt setup.py \
		pi@$(TARGET_HOST):$(NAMESPACE)/$(PKG_NAME)

play: targetonly cleanall test
	$(PKG_NAME) --windows 1 --display $(DISPLAY) --top-margin $(SCREEN_TOP_MARGIN)

play4: targetonly cleanall test
	$(PKG_NAME) --windows 1234 --display $(DISPLAY) --top-margin $(SCREEN_TOP_MARGIN)

replay: targetonly cleanproc
	$(PKG_NAME) --windows 1 --replay

replay4: targetonly cleanproc
	$(PKG_NAME) --windows 1234 --replay

version: targetonly
	$(PKG_NAME) --version

cleanall:
	rm -rf stream/window{1,2,3,4}/*

cleanproc:
	find stream -maxdepth 2 -name "*_processed.png" -type f -delete
