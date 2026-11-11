# RetailPulse: one command per pipeline stage; `make all` reproduces every number from a clean clone.
PY ?= .venv/bin/python
NB = notebooks/01_data_understanding.ipynb notebooks/02_eda.ipynb notebooks/03_stationarity_and_sarima.ipynb \
     notebooks/04_model_benchmark.ipynb notebooks/05_why_analysis.ipynb notebooks/06_forecast_and_inventory.ipynb

.PHONY: all setup data eda train eval why forecast inventory export report notebooks app test clean

all: data eda train eval why forecast inventory export notebooks report test

setup:            ## create the virtual environment and install pinned dependencies
	python3 -m venv .venv && .venv/bin/pip install --upgrade pip && .venv/bin/pip install -r requirements.txt

data:             ## download M5 (no Kaggle login), validate, build weekly series + data profile
	$(PY) -m src.data.download
	$(PY) -m src.data.prepare

eda:              ## EDA figures, stationarity tests, series characteristics, SARIMA deep dive
	$(PY) -m src.analysis.eda
	$(PY) -m src.analysis.sarima_deep_dive

train:            ## all models under the rolling-origin protocol (~45 min on 4 cores)
	$(PY) -m src.models.run_statistical
	$(PY) -m src.models.lightgbm_global
	$(PY) -m src.models.lgbm_ablation

eval:             ## leaderboards, DM tests, bootstrap, coverage, figures
	$(PY) -m src.evaluation.evaluate

why:              ## "why it works here and not there" analysis
	$(PY) -m src.analysis.why

forecast:         ## refit on all data, 13-week forward forecast with intervals
	$(PY) -m src.forecast_final

inventory:        ## safety stock / reorder points + simulated backtest + sensitivity
	$(PY) -m src.inventory.simulate

export:           ## tidy CSVs for Power BI
	$(PY) -m src.export_powerbi

report:           ## render REPORT.md (tables filled from reports/tables/*.csv)
	$(PY) -m scripts.render_report

notebooks:        ## (re)generate and execute all notebooks top-to-bottom in place (outputs are committed)
	$(PY) -m scripts.build_notebooks
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 $(NB)

app:              ## launch the Streamlit dashboard
	$(PY) -m streamlit run app/streamlit_app.py

test:
	$(PY) -m pytest -q tests

clean:
	rm -rf outputs/cache
