<!--
  What: Quick start, how to tweak the assignment, and how to plug in a new predictor.
  When: First time setup or when changing prediction code or algorithms.
-->

# stock-prediction

Forecast stock returns from price/volume history. This repo is set up so you can **swap the built-in moving-average logic** or **add your own model** and run it from the command line.

## Setup

```bash
uv sync
```

Put course data as `sample_data/<name>.npy` (e.g. `s1.npy`).

## Run

```bash
uv run python main.py              # evaluate s1 with the default predictor
uv run python main.py s2           # another file
uv run python main.py --all        # every file in sample_data/
```

Use `--algo` and `--param` once you register a custom predictor (below).

### Check results

- **Terminal** — After a run, look for **`base`**, **`abs`**, and **`rel`** (evaluation metrics). A line like `Predictor: ...` shows which algorithm ran when you used `--algo`.

- **Plots** — Figures are saved under **`output/`**. A single sample run writes something like **`output/01_prediction_eval.png`**. With `--all`, you get one PNG per file, e.g. **`output/prediction_eval_s1.png`**.

- **Full report** — `uv run python main.py --all` prints a summary table and also writes **`output/evaluation_report.txt`**. For numbers only (no PNGs), use `uv run python main.py --all --no-plot` or **`make run`**.

---

## Customize the assignment (no new class)

Edit **`main.py`**:

- **`prediction()`** — your forecast of returns (see `prediction_default()` for the template).
- **`target()`** — usually leave as-is unless the course asks otherwise.

The notebook-style blocks are marked in the file; stay within those areas so grading stays consistent.

---

## Add a new algorithm

1. **Subclass** `Predictor` in **`SES.py`** (or add a new `.py` file next to it).

   Implement `predict(self, P, V, h)` returning a list of forecast returns (same length as `P`).

2. **Register** your module in **`utils.py`**: add the module name to `PREDICTOR_DISCOVERY_MODULES`, e.g. `("SES", "my_model")` if you created `my_model.py`.

3. **Run** using the **class name** (not a short alias):

   ```bash
   uv run python main.py s1 --algo SimpleExponentialSmoothing --param alpha=0.3
   ```

   The built-in moving average is selected with `--algo default`.

4. **Course submission** — update **`authors.md`** / **`explains.md`**, then build the hand-in notebook:

   ```bash
   make submit
   ```

   (Requires `make`, or run `uv run python scripts/submit_notebook.py`.)

   After that, **`Submit.ipynb`** is the file you turn in—upload it to the LMS (or whatever channel the course uses) as your final submission.

---

## Tests

```bash
make test
```

or `uv run python -m unittest discover -s tests -v`.
