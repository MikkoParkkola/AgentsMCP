SHELL := /bin/bash

.PHONY: dev-run dev-run-tui dist clean

dev-run:
	PYTHONPATH=src python -m agentsmcp.cli interactive --no-welcome

dev-run-tui:
	PYTHONPATH=src python -m agentsmcp.cli interactive --ui tui --no-welcome

dist:
	bash scripts/build_dist.sh

clean:
	rm -rf build dist .build build.log

