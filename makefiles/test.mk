# TESTING TARGETS "t"

t test: cfg-audit ## Run fast pytest tests
	pytest -k "not slow"

ta test-all: ## Run all pytest tests
	pytest

cfg-audit: ## Verify Hydra config references
	python scripts/config_audit.py

ca compare-arch: ## Compare medium sized architectures on three epochs
	@echo "=== Training SimpleDenseNet ==="
	python src/train.py trainer.max_epochs=3 tags="[arch_comparison,dense]"
	@echo "=== Training SimpleCNN ==="
	python src/train.py model=mnist_cnn_68k trainer.max_epochs=3 tags="[arch_comparison,cnn]"
	@echo "=== Training ViT ==="
	python src/train.py model=mnist_vit_38k trainer.max_epochs=3 tags="[arch_comparison,vit]"
	@echo "=== Training ConvNeXt-V2 ==="
	python src/train.py model=mnist_convnext_68k trainer.max_epochs=3 tags="[arch_comparison,convnext]"
	@echo "=== Check logs/ directory for results comparison ==="
