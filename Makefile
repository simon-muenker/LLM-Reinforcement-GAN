.PHONY: lint
lint:
	@poetry run ruff check --fix
	@poetry run ruff format
	@poetry run mypy


.PHONY: test
test:
	poetry run pytest -s