# UTILITY TARGETS

# Formatting and linting
f format: ## Run pre-commit hooks
	pre-commit run -a

fp format-preview: ## Preview docformatter actions
	docformatter -c -d -r --black --wrap-summaries=99 --wrap-descriptions=99 --style=sphinx src tests

fdc format-docstrings-check: ## Run docformatter pre-commit hook (manual stage)
	pre-commit run docformatter --hook-stage manual -a

fm format-markdown: ## Run Prettier on Markdown/YAML only
	pre-commit run prettier -a

fn flake8-now: ## Run flake8 lint manually on src/tests and key scripts
	pre-commit run flake8 --hook-stage manual -a

fc format-configs: ## Prettier-format only YAML in configs/
	@FILES=$(shell git ls-files 'configs/**/*.yaml' 'configs/*.yaml'); \
	if [ -n "$$FILES" ]; then \
		pre-commit run prettier --files $$FILES; \
	else \
		echo "No config YAML files found"; \
	fi

# Development utilities
sy sync: ## Merge changes from main branch to your current branch
	git pull
	git pull origin main

tb tensorboard: ## Launch TensorBoard on port 6006
	@lsof -i :6006 >/dev/null 2>&1 && echo "TensorBoard already running on port 6006" || \
		(echo "Starting TensorBoard on port 6006..." && tensorboard --logdir logs/ --reload_interval 1 --port 6006 &)
	@echo "Open http://localhost:6006/"

a activate: ## Activate the uv environment
	@echo "Add to ~/.tcshrc: alias a 'echo \"source .venv/bin/activate.csh\" && source .venv/bin/activate.csh'"
	@echo "Then just type: a"

d deactivate: ## Deactivate the uv environment
	@echo "Add to ~/.tcshrc: alias d 'echo deactivate && deactivate'"
	@echo "Then just type: d"

lc list-configs: ## List available model configurations
	@echo "Available model configs:"
	@find configs/model -name "*.yaml" | sed 's|configs/model/||' | sed 's|\.yaml||' | sort
	@echo "\nAvailable data configs:"
	@find configs/data -name "*.yaml" | sed 's|configs/data/||' | sed 's|\.yaml||' | sort
	@echo "\nAvailable experiment configs:"
	@find configs/experiment -name "*.yaml" | sed 's|configs/experiment/||' | sed 's|\.yaml||' | sort

p po pres presentation:
	(cd docs/presentation && make po)

sep:
	@printf '%*s\n' 180 '' | tr ' ' '+'
