# Lightning Hydra Template Extended - Main Makefile
# All targets are organized into separate include files for better maintainability

# Include all phony declarations
include makefiles/phony.mk

# Include organized sections
include makefiles/clean.mk
include makefiles/display.mk
include makefiles/train-quick.mk
include makefiles/test.mk
include makefiles/experiments.mk
include makefiles/cifar.mk
include makefiles/utils.mk

# Default help target
h help:  ## Show help
	@{ \
	for file in $(MAKEFILE_LIST); do \
		grep -E '^[.a-zA-Z0-9_ -]+:.*?## .*$$' $$file | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'; \
	done; \
	} | less -R
