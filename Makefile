.PHONY: install download features train backtest paper-trade dashboard pipeline visualize test lint

install:
	pip install -r requirements.txt

download:
	python main.py download

features:
	python main.py features

train:
	python main.py train

backtest:
	python main.py backtest

paper-trade:
	python main.py paper-trade --dry-run

dashboard:
	python main.py dashboard

pipeline:
	python main.py pipeline

visualize:
	python main.py visualize

test:
	pytest tests/ -v --tb=short

lint:
	ruff check src/ tests/ main.py
