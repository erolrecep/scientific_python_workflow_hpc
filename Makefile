.PHONY: help all-uv all-mamba env env-uv env-mamba data data-synthetic process process-gpu benchmark benchmark-all test profile clean

# Explicitly set the default goal to 'help'
.DEFAULT_GOAL := help

# Configuration variables
ENV ?= uv
WORKERS ?= 4
MAX_IMAGES ?= 200
BACKEND ?= opencv
DEVICE ?= auto
BATCH_SIZE ?= 64

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
	@echo "Usage: make [command] [ENV=uv|mamba] [WORKERS=N] [BACKEND=opencv|pillow|pytorch]"
	@echo ""
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'
	@echo ""

all-uv: ## Run the entire multi-core workflow using 'uv' as the environment manager
	$(MAKE) ENV=uv process test profile

all-mamba: ## Run the entire multi-core workflow using 'mamba' as the environment manager
	$(MAKE) ENV=mamba process test profile

# Meta-target that routes to the correct environment builder
env: env-$(ENV)

env-uv: ## Create the virtual environment using uv
	@if [ ! -d "$(VENV_DIR)" ]; then \
		uv venv $(VENV_DIR) && \
		uv pip install --python $(PYTHON) -r requirements.txt; \
	fi

env-mamba: ## Create the virtual environment using mamba
	@if [ ! -d "$(VENV_DIR)" ]; then \
		mamba create -y -p $(VENV_DIR) python=3.10 && \
		$(PYTHON) -m pip install -r requirements.txt; \
	fi

data: env ## Download and extract the VGG 17flowers dataset
	$(PYTHON) src/download_data.py

data-synthetic: env ## Generate synthetic test images (instant offline / CI mode)
	$(PYTHON) src/download_data.py --synthetic --num-synthetic $(MAX_IMAGES)

process: env data ## Run the multi-core image processing pipeline across CPU cores
	$(PYTHON) src/process_images.py --workers $(WORKERS) --max-images $(MAX_IMAGES) --backend $(BACKEND)

process-gpu: env data ## Run batched GPU image processing pipeline on CUDA or Apple Silicon MPS
	$(PYTHON) src/process_images.py --backend gpu --max-images $(MAX_IMAGES) --device $(DEVICE) --batch-size $(BATCH_SIZE)

benchmark: env data ## Run 1-core vs multi-core scaling benchmark (supports BACKEND=opencv|pillow|pytorch|gpu|all)
	$(PYTHON) src/process_images.py --benchmark --workers $(WORKERS) --max-images $(MAX_IMAGES) --backend $(BACKEND) --device $(DEVICE) --batch-size $(BATCH_SIZE)

benchmark-all: env data ## Run comparative benchmark across OpenCV, Pillow, PyTorch CPU, and GPU
	$(PYTHON) src/process_images.py --benchmark --workers $(WORKERS) --max-images $(MAX_IMAGES) --backend all --device $(DEVICE) --batch-size $(BATCH_SIZE)

test: env ## Run pytest unit tests to verify image output dimensions and multi-core processing
	$(PYTHON) -m pytest tests/

profile: env data ## Run CPU (Scalene) and Memory (Memray) profilers on the pipeline
	mkdir -p results
	$(PYTHON) -m memray run -f -o results/memray_profile.bin src/process_images.py --workers $(WORKERS) --max-images 50 --no-verify
	$(PYTHON) -m memray flamegraph -f results/memray_profile.bin
	$(PYTHON) -m scalene run --cpu-only -o results/scalene_profile.json src/process_images.py --- --workers $(WORKERS) --max-images 50 --no-verify
	@echo ""
	@echo "======================================================================"
	@echo "✅ PROFILING COMPLETE!"
	@echo "To view the CPU profile in your browser, copy and paste this command:"
	@echo "  $(VENV_DIR)/bin/scalene view results/scalene_profile.json"
	@echo "======================================================================"

clean: ## Remove environments, data downloads, and generated artifacts to reset
	rm -rf .venv ./.mamba-env data/ results/ .pytest_cache/ __pycache__/ src/__pycache__/ tests/__pycache__/
