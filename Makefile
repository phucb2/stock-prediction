# Makefile — common project tasks (use Git Bash or WSL on Windows if `make` is not installed).
.PHONY: run run-all test submit bundle clean help

UV ?= uv
PY := $(UV) run python main.py
UNITTEST := $(UV) run python -m unittest discover -s tests -v
SUBMIT_NB := $(UV) run python scripts/submit_notebook.py
BUNDLE_PY := $(UV) run python scripts/bundle_py.py
CLEAN := $(UV) run python scripts/clean_temp.py

# Evaluate all sample_data/*.npy; print summary and write output/evaluation_report.txt (no PNG charts).
run:
	$(PY) --all --no-plot

# Same as run, but also saves output/prediction_eval_<stem>.png for each file.
run-all:
	$(PY) --all

test:
	$(UNITTEST)

# Build Submit.ipynb: authors.md + explains.md + bundle (SES.py predict from bundle; template stays untouched).
submit:
	$(SUBMIT_NB)

# Project .py except main.py, tests/, scripts/ -> output/py_bundle_manifest.txt + py_bundle.txt
bundle:
	$(BUNDLE_PY)

clean:
	$(CLEAN)

help:
	@echo "make run     - batch eval + report, no plots"
	@echo "make run-all - batch eval + report + per-file PNGs"
	@echo "make test    - run all tests under tests/"
	@echo "make submit  - write Submit.ipynb + verify it runs (see scripts/submit_notebook.py --help)"
	@echo "make bundle  - bundle .py (excludes main.py, utils.py, tests/, scripts/) -> output/py_bundle_*.txt"
	@echo "make clean   - remove output/, Submit.ipynb, __pycache__, *.pyc"
