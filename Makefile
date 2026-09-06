PYTHON ?= python3
VENV   := .venv
PIP    := $(VENV)/bin/pip

.PHONY: help install

help:
	@echo "make install  — create .venv and install requirements.txt"

install:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo ""
	@echo "Dependencies installed successfully!"
	@echo "To activate the virtual environment, run:"
	@echo "source $(VENV)/bin/activate"
