PKG_NAME = tracker
CHECK_FILES = $(PKG_NAME) setup.py
SYNC_FILES = $(PKG_NAME) images Makefile requirements.txt requirements-prod.txt setup.py
PROTO_DIR = $(PKG_NAME)/api/proto
NAMESPACE = ~/Workspace/evercoinx

SOURCE_HOST = fedora
TARGET_HOST = raspberry
EFFECTIVE_HOST = $(shell hostname)

DISPLAY = :10.0
SCREEN_TOP_MARGIN = 98
STREAM_PATH=/mnt/usb1/streams

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

check: sourceonly format imports lint typecheck

format:
	black $(CHECK_FILES)

imports:
	isort $(CHECK_FILES)

lint:
	flake8 $(CHECK_FILES)

typecheck:
	pyright

.PHONY: test
test: targetonly
	python -m unittest discover -s $(PKG_NAME) -p "*_test.py"

installdev: sourceonly
	pip install -r requirements-dev.txt

install: targetonly
	pip install -r requirements-prod.txt
	pip install -e .

deploy: sourceonly
	rsync -avh --delete $(SYNC_FILES) pi@$(TARGET_HOST):$(NAMESPACE)/$(PKG_NAME)

play: targetonly cleanall
	$(PKG_NAME) --windows 0 --display $(DISPLAY) --top-margin $(SCREEN_TOP_MARGIN) --stream-path $(STREAM_PATH)

play4: targetonly cleanall
	$(PKG_NAME) --windows 0123 --display $(DISPLAY) --top-margin $(SCREEN_TOP_MARGIN) --stream-path $(STREAM_PATH)

replay: targetonly cleanproc
	$(PKG_NAME) --windows 0123 --replay --stream-path $(STREAM_PATH)

version: targetonly
	$(PKG_NAME) --version

.PHONY: proto
proto:
	python -m grpc_tools.protoc -I$(PROTO_DIR) --python_out=$(PROTO_DIR) --grpc_python_out=$(PROTO_DIR) \
		$(PROTO_DIR)/analyzer.proto
	sed -i -Ee "s/^import (analyzer_pb2)(.*)/import $(PKG_NAME).api.proto.\1\2/" $(PROTO_DIR)/analyzer_pb2_grpc.py
	black $(PROTO_DIR)
	isort $(PROTO_DIR)

cleanall:
	rm -rf $(STREAM_PATH)/window{0,1,2,3}/*

cleanproc:
	find $(STREAM_PATH) -maxdepth 2 -name "*_processed.png" -type f -delete
