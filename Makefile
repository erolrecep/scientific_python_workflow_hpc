.PHONY: help all-uv all-mamba check-uv check-mamba install-micromamba env env-uv env-mamba data data-synthetic process process-gpu benchmark benchmark-all test profile slides clean

# Ensure user-installed binaries (~/.local/bin and ~/.cargo/bin) are on PATH for all subshells
export PATH := $(HOME)/.local/bin:$(HOME)/.cargo/bin:$(PATH)

# Root prefix for micromamba cache & index (keeps environments isolated without modifying shell profiles)
export MAMBA_ROOT_PREFIX ?= $(HOME)/micromamba

# Detect available mamba binary (micromamba or full mamba)
MAMBA_BIN ?= $(shell command -v micromamba 2>/dev/null || command -v mamba 2>/dev/null || echo micromamba)

# Explicitly set the default goal to 'help'
.DEFAULT_GOAL := help

# Configuration variables
ENV ?= uv
WORKERS ?= 4
MAX_IMAGES ?= 200
BACKEND ?= opencv
DEVICE ?= auto
BATCH_SIZE ?= 64
SLIDES_PORT ?= 8003

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

install-micromamba: ## Explicitly download and install standalone micromamba to ~/.local/bin
	@echo "======================================================================"
	@echo "📦 Installing standalone 'micromamba' into $(HOME)/.local/bin..."
	@echo "ℹ️  Non-interactive, zero-config: does NOT alter shell profiles (~/.bashrc)."
	@echo "======================================================================"
	@mkdir -p $(HOME)/.local/bin
	@if command -v curl >/dev/null 2>&1; then \
		INIT_YES=no CONDA_FORGE_YES=yes BIN_FOLDER="$(HOME)/.local/bin" curl -LsSf https://micro.mamba.pm/install.sh | sh; \
	elif command -v wget >/dev/null 2>&1; then \
		INIT_YES=no CONDA_FORGE_YES=yes BIN_FOLDER="$(HOME)/.local/bin" wget -qO- https://micro.mamba.pm/install.sh | sh; \
	else \
		echo "❌ Error: Neither curl nor wget was found to install micromamba."; \
		exit 1; \
	fi
	@if ! command -v micromamba >/dev/null 2>&1; then \
		echo "❌ Error: micromamba installation finished, but binary was not found in PATH."; \
		echo "👉 Ensure $(HOME)/.local/bin is in your PATH (e.g. in ~/.bashrc or ~/.zshrc)."; \
		exit 1; \
	fi
	@echo "✅ 'micromamba' installed successfully at: $$(which micromamba) ($$(micromamba --version))"
	@echo "======================================================================"

check-mamba: ## Check if mamba or micromamba is installed; auto-installs standalone micromamba if missing
	@if ! command -v mamba >/dev/null 2>&1 && ! command -v micromamba >/dev/null 2>&1; then \
		echo "======================================================================"; \
		echo "⚡ 'mamba' / 'micromamba' package manager not found in PATH."; \
		echo "💡 HPC Tip: If on a cluster, check if modules exist: 'module load miniforge'"; \
		echo "🚀 Auto-installing standalone 'micromamba' into $(HOME)/.local/bin..."; \
		echo "======================================================================"; \
		$(MAKE) install-micromamba; \
	else \
		echo "✅ '$$(basename $$(command -v micromamba 2>/dev/null || command -v mamba 2>/dev/null))' is installed and ready: $$(command -v micromamba 2>/dev/null || command -v mamba 2>/dev/null)"; \
	fi

all-uv: check-uv ## Run the entire multi-core workflow using 'uv' as the environment manager (auto-installs uv if missing)
	$(MAKE) ENV=uv process test profile

all-mamba: check-mamba ## Run the entire multi-core workflow using 'mamba' as the environment manager (auto-installs micromamba if missing)
	$(MAKE) ENV=mamba process test profile

# Meta-target that routes to the correct environment builder
env: env-$(ENV)

env-uv: check-uv ## Create the virtual environment using uv (auto-installs uv if missing)
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "📦 Creating virtual environment $(VENV_DIR) with uv..."; \
		uv venv $(VENV_DIR) && \
		uv pip install --python $(PYTHON) -r requirements.txt; \
	fi

env-mamba: check-mamba ## Create the virtual environment using mamba or micromamba (auto-installs micromamba if missing)
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "📦 Creating virtual environment $(VENV_DIR) with $(MAMBA_BIN)..."; \
		$(MAMBA_BIN) create -y -p $(VENV_DIR) -c conda-forge python=3.10 && \
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

slides: ## Serve the Reveal.js workshop presentation at http://localhost:8003
	@echo "======================================================================"
	@echo "🖥️  Serving Reveal.js Workshop Presentation at http://localhost:$(SLIDES_PORT)"
	@echo "👉 Open your browser to: http://localhost:$(SLIDES_PORT)"
	@echo "👉 Presentation Controls: 'F' for Fullscreen, 'S' for Speaker View, 'ESC' for Map"
	@echo "======================================================================"
	@(cd slides && python3 -m http.server $(SLIDES_PORT))

clean: ## Remove environments, data downloads, and generated artifacts to reset
	rm -rf .venv ./.mamba-env data/ results/ .pytest_cache/ __pycache__/ src/__pycache__/ tests/__pycache__/
