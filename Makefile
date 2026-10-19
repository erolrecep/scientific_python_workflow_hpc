.PHONY: help all-uv all-mamba env env-uv env-mamba data process test profile clean

# Explicitly set the default goal to 'help'
.DEFAULT_GOAL := help

# Set default environment type to uv, but allow override
ENV ?= uv

# Dynamically set paths based on the chosen environment. 
# Mamba requires the ./ prefix to force a local directory installation.
ifeq ($(ENV),mamba)
	VENV_DIR = ./.mamba-env
else
	VENV_DIR = .venv
endif

PYTHON = $(VENV_DIR)/bin/python

help: ## Show this help message and list of available commands
	@echo "======================================================================"
	@echo " Workshop: Managing Scientific Python Environments on HPC"
	@echo "======================================================================"
	@echo "Usage: make [command]"
	@echo ""
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'
	@echo ""

all-uv: ## Run the entire workflow using 'uv' as the environment manager
	$(MAKE) ENV=uv process test profile

all-mamba: ## Run the entire workflow using 'mamba' as the environment manager
	$(MAKE) ENV=mamba process test profile

# Meta-target that routes to the correct environment builder
env: env-$(ENV)

env-uv: ## Create the virtual environment using uv
	uv venv $(VENV_DIR)
	uv pip install --python $(PYTHON) -r requirements.txt

env-mamba: ## Create the virtual environment using mamba
	mamba create -y -p $(VENV_DIR) python=3.10
	$(PYTHON) -m pip install -r requirements.txt

data: ## Download and extract the VGG 17flowers dataset using curl
	mkdir -p data
	@if [ ! -f data/17flowers.tgz ]; then \
		echo "Downloading dataset..."; \
		curl -L -o data/17flowers.tgz https://www.robots.ox.ac.uk/~vgg/data/flowers/17/17flowers.tgz; \
	else \
		echo "Dataset archive already exists. Skipping download."; \
	fi
	tar -xzf data/17flowers.tgz -C data/

process: env data ## Run the image processing pipeline (OpenCV, Pillow, PyTorch)
	$(PYTHON) src/process_images.py

test: env ## Run pytest unit tests to verify image output dimensions
	$(PYTHON) -m pytest tests/

profile: env data ## Run CPU (Scalene) and Memory (Memray) profilers
	mkdir -p results
	$(PYTHON) -m memray run -f -o results/memray_profile.bin src/process_images.py
	$(PYTHON) -m memray flamegraph -f results/memray_profile.bin
	$(PYTHON) -m scalene run --outfile results/scalene_profile.json src/process_images.py
	@echo ""
	@echo "======================================================================"
	@echo "✅ PROFILING COMPLETE!"
	@echo "To view the CPU profile in your browser, copy and paste this command:"
	@echo "  $(VENV_DIR)/bin/scalene view results/scalene_profile.json"
	@echo "======================================================================"

clean: ## Remove environments, data downloads, and generated artifacts to reset
	rm -rf .venv ./.mamba-env data/ results/ .pytest_cache/ __pycache__/ src/__pycache__/ tests/__pycache__/
