<!--
  What: How to run evaluation, pick predictors, submit the notebook, and use helper targets.
  When: Read before running or submitting the stock-prediction assignment.
-->

# stock-prediction

## Layout

| Path | Role |
|------|------|
| `main.py` | Core: load data, `prediction` / `target`, metrics (`evaluate`, `compute_eval_from_pv`). |
| `utils.py` | CLI (`main_cli`), plots, batch report (`output/evaluation_report.txt`), predictor registry and `make_predictor`. |
| `SES.py` | `Predictor` base class and example implementation (`SimpleExponentialSmoothing`). |
| `Sample Assignment.ipynb` | Course template (not overwritten by tooling). |
| `authors.md` | First markdown cell for generated **`Submit.ipynb`** (first segment if split by `---`). |
| `explains.md` | Second markdown cell for **`Submit.ipynb`** (second segment if split by `---`; otherwise whole file). |
| `scripts/submit_notebook.py` | Builds **`Submit.ipynb`**, optionally executes it for a smoke test. |
| `scripts/bundle_py.py` | Writes `output/py_bundle_*.txt` (sources excluding `main.py`, `utils.py`, `tests/`, `scripts/`; strips cross-imports of local modules in the concatenated file). |
| `scripts/clean_temp.py` | Removes generated artifacts (see **`make clean`**). |

Entry point: `uv run python main.py` → delegates to `utils.main_cli()`.

## Setup

From the project root (with [uv](https://github.com/astral-sh/uv) installed):

```bash
uv sync
```

## Data

Place NumPy arrays as `sample_data/<name>.npy` (e.g. `s1.npy`, `s2.npy`).  
Each array should match the course template (columns include close price and volume).

Optional environment variables:

| Variable | Purpose |
|----------|---------|
| `STOCK_DATA_HOME` | Project root; defaults to the repo folder. `sample_data/` lives under it. |
| `STOCK_PV_NPY` | Full path to one `.npy`; when set, single-file runs load this file instead of `sample_data/<stem>.npy`. |

## Run

Use `uv run` so the project environment is used.

### Default predictor (assignment MA template)

```bash
uv run python main.py
```

Evaluates **`sample_data/s1.npy`**, prints `base` / `abs` / `rel`, saves `output/01_prediction_eval.png`.

### Choose algorithm (`--algo` + `--param`)

Predictors are classes subclassing `SES.Predictor`. The special alias **`default`** selects the built-in MA template (`DefaultPredictor`). Any other name must be the **full class name** (case-insensitive), e.g. `SimpleExponentialSmoothing`.

```bash
uv run python main.py s1 --algo SimpleExponentialSmoothing --param alpha=0.3
uv run python main.py --algo default
```

| Flag | Meaning |
|------|---------|
| `--algo NAME` | `default` or a registered class name (e.g. `SimpleExponentialSmoothing`, `DefaultPredictor`). |
| `--param KEY=VALUE` | Keyword passed to the predictor’s `__init__` (repeatable). Values are coerced to int/float/bool when possible. |
| `--predict-h H` | `h` passed to `predict(P, V, h)` (MA window for the default predictor; ignored by SES). |

Short names like `ses` are **not** accepted; use `SimpleExponentialSmoothing`.

New subclasses: define them in a module (e.g. `SES.py`) and add that module to `PREDICTOR_DISCOVERY_MODULES` in `utils.py`.

### One sample file

```bash
uv run python main.py s2
uv run python main.py --sample s2
uv run python main.py --stock MYTICKER
```

The stem is the filename without `.npy` under `sample_data/`.

### All samples and summary report

```bash
uv run python main.py --all
```

- Evaluates every `sample_data/*.npy`.
- Writes `output/prediction_eval_<stem>.png` per file (unless `--no-plot`).
- Writes **`output/evaluation_report.txt`** (includes chosen predictor and `h`).

```bash
uv run python main.py --all --no-plot
```

Console + report only, no PNGs.

Do not combine `--all` with `SOURCE`, `--stock`, or `--sample`.

### Headless (no GUI)

```bash
# PowerShell
$env:MPLBACKEND='Agg'; uv run python main.py --all
```

## Submit notebook (`Submit.ipynb`)

`make submit` runs **`scripts/submit_notebook.py`**. It **reads** **`Sample Assignment.ipynb`** and **writes** **`Submit.ipynb`** (the template file is never modified).

| Input | Notebook cell |
|-------|----------------|
| **`authors.md`** | First cell (markdown): first segment separated by a line containing only `---`, or the whole file. HTML `<!-- ... -->` comments are stripped. |
| **`explains.md`** | Second cell (markdown): second `---` segment, or the whole file if there is only one segment. |
| **`output/py_bundle.txt`** | Regenerated during submit (same as **`make bundle`**: **`scripts/bundle_py.py`**). The full bundle text is inlined into the “# customize your prediction” code cell, then **`_ses = SimpleExponentialSmoothing(...)`** and **`prediction()`** are appended. **`SES.py`** must be in the bundle set. **Data-loading** moves to **index 4**; prediction at **index 3**. |

After writing, the script **executes** the notebook for a smoke test: only cells **through** the `# execute` / `evaluate(p, t, …)` cell (later optional cells are skipped so missing extra tickers do not fail the check).

```bash
make submit
# or
uv run python scripts/submit_notebook.py
```

| Flag | Meaning |
|------|---------|
| `--skip-verify` | Write **`Submit.ipynb`** only; do not execute. |
| `--verify-only` | Execute existing **`Submit.ipynb`** only (no rebuild). |

## Tests

```bash
uv run python -m unittest discover -s tests -v
# or
make test
```

## Customize assignment logic

Edit **`prediction()`** / **`prediction_default()`** in `main.py` (template blocks 3–4–5). Keep **`target()`** aligned with the course definition. Metrics follow the same `evaluate` pairing as the notebook. For SES submission, edit **`SES.py`** and regenerate **`Submit.ipynb`** with **`make submit`**.

## Makefile (optional)

| Target | Command |
|--------|---------|
| `make run` | `--all --no-plot` (report only) |
| `make run-all` | `--all` with per-file PNGs |
| `make test` | Run all tests under `tests/` |
| `make submit` | Build **`Submit.ipynb`** from **`authors.md`**, **`explains.md`**, template; **`SES.py`** must be in the **`scripts/bundle_py.py`** file set; verify execution (see **Submit notebook** above) |
| `make bundle` | **`output/py_bundle_manifest.txt`** + **`output/py_bundle.txt`** (`*.py` except **`main.py`**, **`utils.py`**, **`tests/`**, **`scripts/`**; local project imports removed in the bundle text) |
| `make clean` | Remove **`output/`**, **`Submit.ipynb`**, **`__pycache__`**, **`*.pyc`** (skips `.venv`) |

Requires `make` (e.g. Git Bash or WSL on Windows).
