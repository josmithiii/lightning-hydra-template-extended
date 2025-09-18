# TESTING TARGETS "t"

t test: cfg-audit ## Run fast pytest tests
	pytest -k "not slow"

ta test-all: ## Run all pytest tests
	pytest

cfg-audit: ## Verify Hydra config references
	python scripts/config_audit.py

td test-diagram: ## Generate model architecture diagrams (text + graphical)
	python viz/enhanced_model_diagrams.py

tda test-diagram-all: ## Generate diagrams for all model architectures
	python viz/enhanced_model_diagrams.py -c mnist_cnn_8k
	python viz/enhanced_model_diagrams.py -c mnist_vit_38k
	python viz/enhanced_model_diagrams.py -c mnist_convnext_68k
	python viz/enhanced_model_diagrams.py -c mnist_sdn_8k

tdl test-diagram-list: ## List available model configs for diagrams
	python viz/enhanced_model_diagrams.py --list-configs

tds test-diagram-simple: ## Generate simple text-only diagrams (default mnist_cnn_8k)
	python viz/simple_model_diagram.py

tdsc test-diagram-simple-config: ## Generate simple diagram for specific config (usage: make tdsc CONFIG=mnist_vit_38k)
	python viz/simple_model_diagram.py --config $(CONFIG)

tdsl test-diagram-simple-list: ## List available configs for simple diagrams
	python viz/simple_model_diagram.py --list-configs

tdss test-diagram-simple-samples: ## Generate simple diagrams for sample architectures
	@echo "=== MNIST CNN (8K params) ==="
	python viz/simple_model_diagram.py --config mnist_cnn_8k
	@echo "\n=== MNIST ViT (38K params) ==="
	python viz/simple_model_diagram.py --config mnist_vit_38k
	@echo "\n=== MNIST ConvNeXt (68K params) ==="
	python viz/simple_model_diagram.py --config mnist_convnext_68k
	@echo "\n=== CIFAR-10 CNN (64K params) ==="
	python viz/simple_model_diagram.py --config cifar10_cnn_64k
	@echo "\n=== MNIST Multihead CNN (422K params) ==="
	python viz/simple_model_diagram.py --config mnist_mh_cnn_422k

tdsm test-diagram-simple-mnist: ## Generate simple diagrams for all MNIST architectures
	@echo "=== MNIST Architectures ==="
	python viz/simple_model_diagram.py --config mnist_cnn_8k
	python viz/simple_model_diagram.py --config mnist_sdn_8k
	python viz/simple_model_diagram.py --config mnist_vit_38k
	python viz/simple_model_diagram.py --config mnist_convnext_68k
	python viz/simple_model_diagram.py --config mnist_mh_cnn_422k

tdsc10 test-diagram-simple-cifar10: ## Generate simple diagrams for CIFAR-10 architectures
	@echo "=== CIFAR-10 Architectures ==="
	python viz/simple_model_diagram.py --config cifar10_cnn_64k
	python viz/simple_model_diagram.py --config cifar10_convnext_64k
	python viz/simple_model_diagram.py --config cifar10_mh_cnn_64k
	python viz/simple_model_diagram.py --config cifar10_vit_210k

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