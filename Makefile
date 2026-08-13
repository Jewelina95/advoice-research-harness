PYTHON ?= /Users/wenshaoyue/anaconda3/bin/python
DATASET ?= NCMMSC2021_AD

.PHONY: validate quick full dataset report test clean-cache

validate:
	PYTHONPATH=src $(PYTHON) -m advoice validate --dataset $(DATASET)

quick:
	PYTHONPATH=src $(PYTHON) -m advoice run --dataset $(DATASET) --mode quick --agent-provider disabled

full:
	PYTHONPATH=src $(PYTHON) -m advoice run --dataset $(DATASET) --mode full --agent-provider codex_cli

dataset:
	PYTHONPATH=src $(PYTHON) -m advoice run --dataset $(DATASET) --mode full --agent-provider codex_cli

report:
	PYTHONPATH=src $(PYTHON) -m advoice report --dataset $(DATASET)

test:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src $(PYTHON) -m pytest -q

clean-cache:
	PYTHONPATH=src $(PYTHON) -m advoice clean-cache --dataset $(DATASET)
