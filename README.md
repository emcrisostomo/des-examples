# des-examples

Dependency management uses [uv](https://github.com/astral-sh/uv).

## Common commands

- `make create-venv` creates `.venv` with `uv venv`.
- `make dependencies-compile` compiles `requirements.in` to `requirements.txt` with `uv pip compile`.
- `make dependencies-install` syncs `.venv` from `requirements.txt` with `uv pip sync`.
- `make requirements-update` refreshes pinned versions in `requirements.txt`.
