# Phony target declarations - organized by category

# Help and utilities
.PHONY: h help

# Cleaning targets
.PHONY: c clean dc dclean cl clean-logs

# Display targets
.PHONY: dv display-vimh

# Quick training targets
.PHONY: tq train-quick tqc train-quick-cnn tqv train-quick-vit tqcn train-quick-convnext tqvh train-quick-vimh tqa train-quick-all

# Testing targets
.PHONY: t test ta test-all cfg-audit

# Diagram targets
.PHONY: td test-diagram tda test-diagram-all tdl test-diagram-list tds test-diagram-simple tdsc test-diagram-simple-config tdsl test-diagram-simple-list tdss test-diagram-simple-samples tdsm test-diagram-simple-mnist tdsc10 test-diagram-simple-cifar10

# Architecture comparison
.PHONY: ca compare-arch

# Experiment targets
.PHONY: e esdn exp-sdn ec10c exp-cifar10-cnn evit exp-vit ev995 exp-vit-995 ecm exp-cnn-mnist emhcm exp-multihead-cnn-mnist emhcc10 exp-multihead-cnn-cifar10 evimh exp-vimh-16kdss evimho exp-vimh-16kdss-ordinal evimhr exp-vimh-16kdss-regression excn exp-convnext ecnb exp-convnext-benchmark

# CIFAR-10 benchmarks
.PHONY: cb10c cifar10-cnn cb10cn cifar10-convnext cb10cn64 cifar10-convnext-64k-optimized cb10cn128 cifar10-convnext-128k-optimized cb10v cifar10-vit cb10e cifar10-efficientnet

# CIFAR-100 benchmarks
.PHONY: cb100c cifar100-cnn cb100cmh cifar100-cnn-multihead cb100cn cifar100-convnext cb100v cifar100-vit cb100e cifar100-efficientnet cb100sdn cifar100-sdn cb100cnn1m cifar100-cnn-1m cb100cn1m cifar100-convnext-1m cb100cn10m cifar100-convnext-10m cb100cc cifar100-coarse-cnn cb100ccn cifar100-coarse-convnext

# Quick CIFAR benchmarks
.PHONY: cbq10c cifar10-quick-cnn cbq10cn cifar10-quick-convnext cbq10cn64 cifar10-quick-convnext-64k cbq10cn128 cifar10-quick-convnext-128k cbq100c cifar100-quick-cnn cbq100sdn cifar100-quick-sdn cbq100cnn1m cifar100-quick-cnn-1m cbq100cn1m cifar100-quick-convnext-1m cbq100cn10m cifar100-quick-convnext-10m cbq100cc cifar100-quick-coarse cbqa cifar-benchmark-quick-all

# Benchmark suites
.PHONY: cbs benchmark-suite cbs10 benchmark-suite-cifar10 cbs100 benchmark-suite-cifar100 cbsa cifar-benchmark-suite-all allqt all-quick-tests

# Formatting and linting
.PHONY: f format fp format-preview fdc format-docstrings-check fm format-markdown fn flake8-now fc format-configs

# Utilities
.PHONY: sy sync tb tensorboard a activate d deactivate lc list-configs p po pres presentation sep