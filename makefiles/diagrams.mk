# TEXT DIAGRAM TARGETS "td"

td text-diagram: ## Generate model architecture diagrams (text + graphical)
	python viz/enhanced_model_diagrams.py

tda text-diagram-all: ## Generate diagrams for all model architectures
	python viz/enhanced_model_diagrams.py -c mnist_cnn_8k
	python viz/enhanced_model_diagrams.py -c mnist_vit_38k
	python viz/enhanced_model_diagrams.py -c mnist_convnext_68k
	python viz/enhanced_model_diagrams.py -c mnist_sdn_8k

tdl text-diagram-list: ## List available model configs for diagrams
	python viz/enhanced_model_diagrams.py --list-configs

tds text-diagram-simple: ## Generate simple text-only diagrams (default mnist_cnn_8k)
	python viz/simple_model_diagram.py

tdsc text-diagram-simple-config: ## Generate simple diagram for specific config (usage: make tdsc CONFIG=mnist_vit_38k)
	python viz/simple_model_diagram.py --config $(CONFIG)

tdsl text-diagram-simple-list: ## List available configs for simple diagrams
	python viz/simple_model_diagram.py --list-configs

tdss text-diagram-simple-samples: ## Generate simple diagrams for sample architectures
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

tdsm text-diagram-simple-mnist: ## Generate simple diagrams for all MNIST architectures
	@echo "=== MNIST Architectures ==="
	python viz/simple_model_diagram.py --config mnist_cnn_8k
	python viz/simple_model_diagram.py --config mnist_sdn_8k
	python viz/simple_model_diagram.py --config mnist_vit_38k
	python viz/simple_model_diagram.py --config mnist_convnext_68k
	python viz/simple_model_diagram.py --config mnist_mh_cnn_422k

tdsc10 text-diagram-simple-cifar10: ## Generate simple diagrams for CIFAR-10 architectures
	@echo "=== CIFAR-10 Architectures ==="
	python viz/simple_model_diagram.py --config cifar10_cnn_64k
	python viz/simple_model_diagram.py --config cifar10_convnext_64k
	python viz/simple_model_diagram.py --config cifar10_mh_cnn_64k
	python viz/simple_model_diagram.py --config cifar10_vit_210k

