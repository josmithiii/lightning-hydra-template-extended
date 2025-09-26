# TRAIN-QUICKLY TARGETS "tq"

tq train-quick: check-env ## Train quickly on the defaults (SimpleDenseNet on MNIST), 1 epoch
	python src/train.py trainer.max_epochs=1 +trainer.limit_train_batches=10 +trainer.limit_val_batches=5

tqc train-quick-cnn: check-env ## Train quickly SimpleCNN, 1 epoch
	python src/train.py model=mnist_cnn_8k trainer.max_epochs=1 +trainer.limit_train_batches=10 +trainer.limit_val_batches=5

tqv train-quick-vit: check-env ## Train quickly ViT, 1 epoch
	python src/train.py model=mnist_vit_38k trainer.max_epochs=1 +trainer.limit_train_batches=10 +trainer.limit_val_batches=5

tqcn train-quick-convnext: check-env ## Train quickly ConvNeXt-V2, 1 epoch
	python src/train.py model=mnist_convnext_68k trainer.max_epochs=1 +trainer.limit_train_batches=10 +trainer.limit_val_batches=5

tqvh train-quick-vimh: check-env ## Train quickly SimpleCNN on VIMH examples dataset, 1 epoch
	python src/train.py experiment=vimh_cnn trainer.max_epochs=1 +trainer.limit_train_batches=10 +trainer.limit_val_batches=5

tqa train-quick-all: tq sep tqc sep tqv sep tqcn sep tqvh ## Train quickly all architectures supported on mnist and cnn on vimh