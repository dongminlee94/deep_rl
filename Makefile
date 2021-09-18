format:
		black . --line-length 104
		isort .

lint:
		env PYTHONPATH=. pytest . --pylint --flake8 --mypy

setup:
		pip install -r requirements.txt
		pre-commit install