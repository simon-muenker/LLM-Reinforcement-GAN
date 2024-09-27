.PHONY: pretty
pretty:
	@poetry run ruff check --fix
	@poetry run ruff format


.PHONY: lint
lint:
	@poetry run mypy


.PHONY: test
test:
	@poetry run pytest -s


.PHONY: experiment_twon
experiment_twon:
	@poetry run python ./experiments/twon/experiment.py
