# CIFAR BENCHMARKS "cb" - Computer Vision Dataset Experiments

# CIFAR-10 benchmarks
cb10c cifar10-cnn: ## Run CIFAR-10 CNN benchmark (85-92% expected accuracy)
	time python src/train.py experiment=cifar10_benchmark_cnn

cb10cn cifar10-convnext: ## Run CIFAR-10 ConvNeXt benchmark (90-95% expected accuracy)
	time python src/train.py experiment=cifar10_benchmark_convnext

cb10cn64 cifar10-convnext-64k-optimized: ## Run CIFAR-10 ConvNeXt 64K optimized for small images
	time python src/train.py experiment=cifar10_convnext_64k_optimized

cb10cn128 cifar10-convnext-128k-optimized: ## Run CIFAR-10 ConvNeXt 128K optimized for small images
	time python src/train.py experiment=cifar10_convnext_128k_optimized

cb10v cifar10-vit: ## Run CIFAR-10 Vision Transformer benchmark (88-93% expected accuracy)
	time python src/train.py experiment=cifar10_benchmark_vit

cb10e cifar10-efficientnet: ## Run CIFAR-10 EfficientNet benchmark (89-94% expected accuracy)
	time python src/train.py experiment=cifar10_benchmark_efficientnet

# CIFAR-100 benchmarks
cb100c cifar100-cnn: ## Run CIFAR-100 CNN benchmark (55-70% expected accuracy) [not MPS compatible]
	time python src/train.py experiment=cifar100_benchmark_cnn

cb100cmh cifar100-cnn-multihead: ## Run CIFAR-100 CNN benchmark using multihead classifier
	time python src/train.py experiment=cifar100mh_cnn

cb100cn cifar100-convnext: ## Run CIFAR-100 ConvNeXt benchmark (70-80% expected accuracy)
	time python src/train.py experiment=cifar100_benchmark_convnext

cb100v cifar100-vit: ## Run CIFAR-100 Vision Transformer benchmark (65-75% expected accuracy)
	time python src/train.py experiment=cifar100_vit_210k

cb100e cifar100-efficientnet: ## Run CIFAR-100 EfficientNet benchmark (68-78% expected accuracy)
	time python src/train.py experiment=cifar100_efficientnet_210k

cb100sdn cifar100-sdn: ## Run CIFAR-100 SimpleDenseNet benchmark (~1M params, 50-65% expected accuracy)
	time python src/train.py model=cifar100_sdn_1m data=cifar100

cb100cnn1m cifar100-cnn-1m: ## Run CIFAR-100 CNN benchmark (~1M params, 55-70% expected accuracy)
	time python src/train.py model=cifar100_cnn_1m data=cifar100

cb100cn1m cifar100-convnext-1m: ## Run CIFAR-100 ConvNeXt benchmark (~1M params, 70-80% expected accuracy)
	time python src/train.py model=cifar100_convnext_1m data=cifar100

cb100cn10m cifar100-convnext-10m: ## Run CIFAR-100 ConvNeXt 10M benchmark (~10M params, 65-75% expected accuracy)
	time python src/train.py model=cifar100_convnext_10m data=cifar100

cb100cc cifar100-coarse-cnn: ## Run CIFAR-100 coarse (20-class) CNN benchmark (75-85% expected accuracy) - CPU required due to MPS pooling limitation (and no GPU here)
	time python src/train.py experiment=cifar100_coarse_cnn trainer=cpu

cb100ccn cifar100-coarse-convnext: ## Run CIFAR-100 coarse ConvNeXt benchmark (80-90% expected accuracy)
	time python src/train.py experiment=cifar100_coarse_convnext

# CIFAR QUICK BENCHMARKS "cbq" - Fast Validation Runs
cbq10c cifar10-quick-cnn: ## Quick CIFAR-10 CNN validation (5 epochs)
	python src/train.py experiment=cifar10_benchmark_cnn trainer.max_epochs=5 trainer.min_epochs=1

cbq10cn cifar10-quick-convnext: ## Quick CIFAR-10 ConvNeXt validation (5 epochs)
	python src/train.py experiment=cifar10_benchmark_convnext trainer.max_epochs=5 trainer.min_epochs=1

cbq10cn64 cifar10-quick-convnext-64k: ## Quick CIFAR-10 ConvNeXt 64K optimized validation (5 epochs)
	python src/train.py experiment=cifar10_convnext_64k_optimized trainer.max_epochs=5 trainer.min_epochs=1

cbq10cn128 cifar10-quick-convnext-128k: ## Quick CIFAR-10 ConvNeXt 128K optimized validation (5 epochs)
	python src/train.py experiment=cifar10_convnext_128k_optimized trainer.max_epochs=5 trainer.min_epochs=1

cbq100c cifar100-quick-cnn: ## Quick CIFAR-100 CNN validation (5 epochs)
	python src/train.py experiment=cifar100_benchmark_cnn trainer.max_epochs=5 trainer.min_epochs=1

cbq100sdn cifar100-quick-sdn: ## Quick CIFAR-100 SimpleDenseNet validation (5 epochs)
	python src/train.py model=cifar100_sdn_1m data=cifar100 trainer.max_epochs=5 trainer.min_epochs=1

cbq100cnn1m cifar100-quick-cnn-1m: ## Quick CIFAR-100 CNN 1M validation (5 epochs)
	python src/train.py model=cifar100_cnn_1m data=cifar100 trainer.max_epochs=5 trainer.min_epochs=1

cbq100cn1m cifar100-quick-convnext-1m: ## Quick CIFAR-100 ConvNeXt 1M validation (5 epochs)
	python src/train.py model=cifar100_convnext_1m data=cifar100 trainer.max_epochs=5 trainer.min_epochs=1

cbq100cn10m cifar100-quick-convnext-10m: ## Quick CIFAR-100 ConvNeXt 10M validation (10 epochs)
	python src/train.py model=cifar100_convnext_10m data=cifar100 trainer.max_epochs=10 trainer.min_epochs=5

cbq100cc cifar100-quick-coarse: ## Quick CIFAR-100 coarse validation (5 epochs)
	python src/train.py experiment=cifar100_coarse_cnn trainer.max_epochs=5 trainer.min_epochs=1

cbqa cifar-benchmark-quick-all: cbq10c sep cbq10cn sep cbq10cn64 sep cbq10cn128 sep cbq100sdn sep cbq100cnn1m sep cbq100cn10m sep cbq100c sep cbq100cc  ## Run all quick CIFAR validations

# CIFAR BENCHMARK SUITES "cbs" - Systematic Comparisons
cbs benchmark-suite: ## Run automated CIFAR benchmark suite
	python benchmarks/scripts/benchmark_cifar.py

cbs10 benchmark-suite-cifar10: cb10c sep cb10cn sep cb10v sep cb10e ## Run all CIFAR-10 benchmarks
	@echo "=== CIFAR-10 benchmark suite complete ==="

cbs100 benchmark-suite-cifar100: cb100c sep cb100cn sep cb100v sep cb100e sep cb100sdn sep cb100cnn1m sep cb100cn1m sep cb100cn10m sep cb100cc sep cb100ccn ## Run all CIFAR-100 benchmarks
	@echo "=== CIFAR-100 benchmark suite complete ==="

cbsa cifar-benchmark-suite-all: cbs10 cbs100 ## Run complete CIFAR benchmark suite
	@echo "=== Complete CIFAR benchmark suite finished ==="

allqt all-quick-tests: tqa cbqa ## All quick tests