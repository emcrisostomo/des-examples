UV ?= uv
UV_CACHE_DIR ?= /tmp/.uv-cache
VENV_ROOT ?= .venv
VENV_PATH := $(VENV_ROOT)/bin
VENV_PYTHON := $(VENV_PATH)/python
UV_RUN = UV_CACHE_DIR=$(UV_CACHE_DIR) $(UV)
PY_SCRIPTS := \
	2-way-mirror-mttf.py \
	3-way-mirror-mttf.py \
	aws_des_availability.py \
	aws_des_availability_simpy.py \
	aws_des_availability_simpy_idiomatic.py \
	convolution.py \
	des.py \
	markov.py \
	shard_overlap_simulation.py
RUN_TARGETS := $(patsubst %.py,run-%,$(PY_SCRIPTS))

.PHONY: create-venv
default: dependencies-install

create-venv:
	[ -x "$(VENV_PYTHON)" ] || $(UV_RUN) venv $(VENV_ROOT)

requirements.txt: requirements.in | create-venv
	$(UV_RUN) pip compile --python $(VENV_PYTHON) requirements.in -o requirements.txt

.PHONY: dependencies-compile
dependencies-compile: requirements.txt

.PHONY: dependencies-install
dependencies-install: requirements.txt | create-venv
	$(UV_RUN) pip sync --python $(VENV_PYTHON) requirements.txt

.PHONY: requirements-update
requirements-update: | create-venv
	$(UV_RUN) pip compile --upgrade --python $(VENV_PYTHON) requirements.in -o requirements.txt

.PHONY: $(RUN_TARGETS)
$(RUN_TARGETS): run-%: dependencies-install
	$(VENV_PYTHON) $*.py
