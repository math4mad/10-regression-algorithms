# Python + Quarto helpers for the 10-regression-algorithms note.
# JUPYTER_PATH makes Quarto resolve the `python3` kernel from .venv rather than from a
# stale user-level kernelspec in ~/Library/Jupyter (or ~/.local/share/jupyter).
VENV := .venv
PY   := $(VENV)/bin/python
export JUPYTER_PATH := $(abspath $(VENV)/share/jupyter)
export PATH         := $(abspath $(VENV)/bin):$(PATH)
# reg10 is a plain source package (no install step), so it must be on the path.
export PYTHONPATH   := $(abspath src)

.PHONY: venv install smoke compare pairs render preview publish clean

venv:
	python3 -m venv $(VENV)

install: venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

smoke:             ## package smoke test: fit all ten, print the RMSD table
	$(PY) -W ignore -m reg10.models

compare:           ## port of 10-algorithm-compare.jl
	$(PY) scripts/compare_algorithms.py

pairs:             ## port of cor-plot.jl
	$(PY) scripts/pair_plot.py

render:            ## execute index.qmd and build the site into _site/
	quarto render index.qmd

preview:
	quarto preview index.qmd

publish:           ## publishing runs in GitHub Actions -- push the branch to trigger it
	git push origin HEAD

clean:
	rm -rf _site .quarto figures
	rm -rf __pycache__ src/reg10/__pycache__ scripts/__pycache__
