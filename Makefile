# Blueberry LLM helper targets

PYTHON ?= python
WANDB_MODE ?= offline

.PHONY: help setup auto-config train-auto cli-run demo-optimizers docs clean-artifacts

help:
	@echo "Available targets:"
	@echo "  make setup             # Install dependencies via setup.sh"
	@echo "  make auto-config       # Print detected hardware + suggested config"
	@echo "  make train-auto        # Run auto-configured training script"
	@echo "  make cli-run           # Launch a single CLI experiment (offline wandb)"
	@echo "  make demo-optimizers   # Run optimizer comparison grid with artifacts"
	@echo "  make docs              # Build MkDocs site"
	@echo "  make clean-artifacts   # Remove generated run artifacts (careful)"

setup:
	chmod +x setup.sh
	./setup.sh

auto-config:
	$(PYTHON) auto_config.py

train-auto:
	$(PYTHON) train_auto.py

cli-run:
	WANDB_MODE=$(WANDB_MODE) $(PYTHON) -m blueberry_cli run \
		--experiment-id make_cli_run \
		--max-steps 50 \
		--batch-size 16 \
		--collect-router-stats \
		--out-dir experiments/runs \
		--notes "make cli-run" \
		--optimizer adamw \
		--learning-rate 0.001

demo-optimizers:
	WANDB_MODE=$(WANDB_MODE) $(PYTHON) -m blueberry_cli grid \
		--experiment-id demo_opt \
		--regime flops \
		--experts 1,4 \
		--top-k 2 \
		--max-steps 60 \
		--batch-size 12 \
		--collect-router-stats \
		--optimizers adamw,muon \
		--adamw-lr 0.001 \
		--muon-lr 0.01 \
		--seeds 1 \
		--eval-steps 30 \
		--out-dir experiments/runs \
		--tag demo --notes "optimizer sweep demo"

docs:
	$(PYTHON) -m mkdocs build

clean-artifacts:
	rm -rf experiments/runs/* data_cache/*
