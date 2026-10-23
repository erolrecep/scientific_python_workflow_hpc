.PHONY: help all-uv all-mamba check-uv check-mamba env env-uv env-mamba data data-synthetic process process-gpu benchmark benchmark-all test profile clean

# Ensure user-installed binaries (~/.local/bin and ~/.cargo/bin) are on PATH for all subshells
export PATH := $(HOME)/.local/bin:$(HOME)/.cargo/bin:$(PATH)

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
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'
	@echo ""

check-uv: ## Check if uv is installed; if missing, automatically download and install it
	@if ! command -v uv >/dev/null 2>&1; then \
		echo "======================================================================"; \
		echo "⚡ 'uv' package manager not found in PATH."; \
		echo "📦 'uv' is an ultra-fast Python environment manager required for this workshop."; \
		echo "🚀 Automatically installing uv via official Astral installer..."; \
		echo "======================================================================"; \
		if command -v curl >/dev/null 2>&1; then \
			curl -LsSf https://astral.sh/uv/install.sh | sh; \
		elif command -v wget >/dev/null 2>&1; then \
			wget -qO- https://astral.sh/uv/install.sh | sh; \
		elif command -v python3 >/dev/null 2>&1; then \
			python3 -m pip install --user uv; \
		else \
			echo "❌ Error: Neither curl, wget, nor python3 were found to install uv."; \
			echo "👉 Please install uv manually: https://docs.astral.sh/uv/"; \
			exit 1; \
		fi; \
		if ! command -v uv >/dev/null 2>&1; then \
			echo "❌ Error: uv installation finished, but 'uv' binary could not be found in PATH."; \
			echo "👉 Ensure $(HOME)/.local/bin is added to your shell PATH (e.g. in ~/.bashrc or ~/.zshrc)."; \
			exit 1; \
		fi; \
		echo "✅ 'uv' installed successfully at: $$(which uv) ($$(uv --version))"; \
		echo "======================================================================"; \
	else \
		echo "✅ 'uv' is installed and ready: $$(which uv) ($$(uv --version))"; \
	fi

check-mamba: ## Check if mamba or micromamba is installed
	@if ! command -v mamba >/dev/null 2>&1 && ! command -v micromamba >/dev/null 2>&1; then \
		echo "======================================================================"; \
		echo "⚠️  'mamba' or 'micromamba' not found in PATH."; \
		echo "👉 To use Mamba, install Miniforge: https://github.com/conda-forge/miniforge"; \
		echo "💡 Or simply use 'make all-uv' for instant zero-config Python speed!"; \
		echo "======================================================================"; \
		exit 1; \
	else \
		echo "✅ 'mamba'/'micromamba' is installed and ready: $$(which mamba 2>/dev/null || which micromamba 2>/dev/null)"; \
	fi

all-uv: check-uv ## Run the entire multi-core workflow using 'uv' as the environment manager (auto-installs uv if missing)
	$(MAKE) ENV=uv process test profile

all-mamba: check-mamba ## Run the entire multi-core workflow using 'mamba' as the environment manager
	$(MAKE) ENV=mamba process test profile

# Meta-target that routes to the correct environment builder
env: env-$(ENV)

env-uv: check-uv ## Create the virtual environment using uv (auto-installs uv if missing)
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "📦 Creating virtual environment $(VENV_DIR) with uv..."; \
		uv venv $(VENV_DIR) && \
		uv pip install --python $(PYTHON) -r requirements.txt; \
	fi

env-mamba: check-mamba ## Create the virtual environment using mamba
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
