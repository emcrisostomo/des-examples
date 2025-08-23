VENV_PATH = venv/bin
PIP_COMPILE = $(VENV_PATH)/pip-compile --strip-extras -r requirements.in

.PHONY: dependencies-compile
dependencies-compile: requirements.txt

requirements.txt: requirements.in
	$(PIP_COMPILE)

.PHONY: dependencies-install
dependencies-install: dependencies-compile
	$(VENV_PATH)/pip install -r requirements.txt

.PHONY: requirements-update
requirements-update:
	$(PIP_COMPILE)
