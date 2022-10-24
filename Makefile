ALL_NBS_SOURCE := $(shell echo *.ipynb)
ALL_NBS_DOC_TARGETS := $(addprefix docs/,$(ALL_NBS_SOURCE))

$(info $(ALL_NBS_DOC_TARGETS))

docs/%.ipynb: %.ipynb
	poetry run python change_renderer.py $< > $@


all-docs-ipynb: $(ALL_NBS_DOC_TARGETS)

.PHONY: all-docs-ipynb


all-docs: all-docs-ipynb
	poetry run mkdocs build -d site

.PHONY: all-docs