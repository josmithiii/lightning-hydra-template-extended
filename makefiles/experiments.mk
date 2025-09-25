# EXPERIMENTS "e" - Reproducible Configuration Examples

e esdn exp-sdn: ## Run original example experiment (reproducible baseline)
	time python src/train.py experiment=example

ec10c exp-cifar10-cnn: ## Run CIFAR-10 CNN benchmark (85-92% expected accuracy)
	time python src/train.py experiment=cifar10_cnn_cpu

evit exp-vit: ## Run ViT experiment
	time python src/train.py experiment=vit_mnist

ev995 exp-vit-995: ## Run ViT experiment achieving SOTA 99.5% validation accuracy
	time python src/train.py experiment=vit_mnist_995
# == python src/train.py model=mnist_vit_995 data=mnist_vit_995
#           trainer.max_epochs=200 trainer.min_epochs=10
#           trainer.gradient_clip_val=1.0 data.batch_size=128 seed=12345
#           tags="[mnist,vit,995,optimized]"

ecm exp-cnn-mnist: ## Run single-head CNN MNIST classification experiment - accuracy ~99.1%
	time python src/train.py experiment=cnn_mnist

emhcm exp-multihead-cnn-mnist: ## Run MultiHead CNN MNIST classification experiment - accuracies ~99.1%, 99.2%, 99.2%
	time python src/train.py experiment=multihead_cnn_mnist

emhcc10 exp-multihead-cnn-cifar10: ## Run MultiHead CNN CIFAR-10 classification experiment
	time python src/train.py experiment=multihead_cnn_cifar10

evimh exp-vimh-16kdss: ## Run VIMH CNN training with 16K dataset samples (SimpleSynth)
	time python src/train.py experiment=vimh_cnn_16kdss # ./configs/experiment/vimh_cnn_16kdss.yaml

evimho exp-vimh-16kdss-ordinal: ## Run VIMH CNN training with ordinal regression loss (distance-aware)
	time python src/train.py experiment=vimh_cnn_16kdss_ordinal # ./configs/experiment/vimh_cnn_16kdss.yaml

evimhr exp-vimh-16kdss-regression: ## Run VIMH CNN training with pure regression heads (sigmoid + parameter mapping)
	time python src/train.py experiment=vimh_cnn_16kdss_regression # ./configs/experiment/vimh_cnn_16kdss_regression.yaml

evimhst exp-vimh-16kdss-soft-target: ## Run VIMH CNN training with soft target loss (smooth probability distributions)
	time python src/train.py experiment=vimh_cnn_16kdss_soft_target

evimhwce exp-vimh-16kdss-weighted-ce: ## Run VIMH CNN training with weighted cross entropy loss (distance-based penalties)
	time python src/train.py experiment=vimh_cnn_16kdss_weighted_ce

evimhqr exp-vimh-16kdss-quantized-regression: ## Run VIMH CNN training with quantized regression loss (direct continuous prediction)
	time python src/train.py experiment=vimh_cnn_16kdss_quantized_regression

excn exp-convnext: ## Run ConvNeXt-V2 experiment
	time python src/train.py experiment=convnext_mnist

ecnb exp-convnext-benchmark: ## Run official ConvNeXt V2-Tiny benchmark (acid test)
	time python src/train.py experiment=convnext_v2_official_tiny_benchmark

ea exp-all: ## Run ALL experiments, capturing outputs in experiment_logs/
	time bash scripts/run_all_experiments.sh --force

en exp-new: ## Run all new experiments not having a log yet, capturing their outputs in experiment_logs/
	time bash scripts/run_all_experiments.sh --jobs 1

exp-clean: ## Clean all experiment logs in ./experiment_logs/
	/bin/rm -rf ./experiment_logs/

xl extract-logs:
	python ./scripts/extract_logs.py

xlu extract-logs-update:
	python ./scripts/extract_logs.py --csv > experiments_overview.md
