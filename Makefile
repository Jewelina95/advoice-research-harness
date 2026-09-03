PYTHON ?= .venv/bin/python
DATASET ?= NCMMSC2021_AD

.PHONY: validate quick full all all-full prepare-release-gate prepare-audit processed processed-all dataset evaluate evaluate-all report aggregate demo demo-sample test clean-cache

validate:
	PYTHONPATH=src $(PYTHON) -m advoice validate --dataset $(DATASET)

quick:
	PYTHONPATH=src $(PYTHON) -m advoice run --dataset $(DATASET) --mode quick --agent-provider disabled

full:
	PYTHONPATH=src $(PYTHON) -m advoice run --dataset $(DATASET) --mode full --agent-provider openai_api

all:
	PYTHONPATH=src $(PYTHON) -m advoice run-all --mode quick --agent-provider disabled

all-full: prepare-release-gate
	PYTHONPATH=src $(PYTHON) -m advoice run-all --mode full --agent-provider openai_api

prepare-release-gate:
	PYTHONPATH=src $(PYTHON) -m advoice run --dataset PREPARE_DrivenData --mode full --agent-provider openai_api
	PYTHONPATH=src $(PYTHON) -m advoice evaluate --dataset PREPARE_DrivenData
	PYTHONPATH=src $(PYTHON) scripts/check_prepare_speechcare_gate.py
	PYTHONPATH=src $(PYTHON) scripts/build_prepare_9_2_method_audit_report.py
	PYTHONPATH=src $(PYTHON) scripts/check_prepare_speechcare_gate.py --enforce

prepare-audit:
	PYTHONPATH=src $(PYTHON) scripts/check_prepare_speechcare_gate.py
	PYTHONPATH=src $(PYTHON) scripts/build_prepare_9_2_method_audit_report.py

processed:
	PYTHONPATH=src $(PYTHON) -m advoice run-processed --dataset $(DATASET) --agent-provider disabled

processed-all:
	PYTHONPATH=src $(PYTHON) -m advoice run-all-processed --agent-provider disabled

dataset:
	PYTHONPATH=src $(PYTHON) -m advoice run --dataset $(DATASET) --mode full --agent-provider openai_api

evaluate:
	PYTHONPATH=src $(PYTHON) -m advoice evaluate --dataset $(DATASET)

evaluate-all:
	PYTHONPATH=src $(PYTHON) -m advoice evaluate-all

report:
	PYTHONPATH=src $(PYTHON) -m advoice report --dataset $(DATASET)

aggregate:
	PYTHONPATH=src $(PYTHON) -m advoice aggregate-report

demo-sample:
	PYTHONPATH=src $(PYTHON) demo/generate_sample.py
	PYTHONPATH=src $(PYTHON) demo/run_demo.py

demo: demo-sample
	PYTHONPATH=src $(PYTHON) demo/server.py

test:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src $(PYTHON) -m pytest -q

clean-cache:
	PYTHONPATH=src $(PYTHON) -m advoice clean-cache --dataset $(DATASET)
