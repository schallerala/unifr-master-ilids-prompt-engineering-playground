ALL_NBS_SOURCE := $(shell echo *.ipynb)
ALL_NBS_DOC_TARGETS := $(addprefix docs/,$(ALL_NBS_SOURCE))

docs/%.ipynb: %.ipynb mkdocs.yml
	cp $< $@


all-docs-ipynb: $(ALL_NBS_DOC_TARGETS)

.PHONY: all-docs-ipynb


all-docs: all-docs-ipynb
	poetry run mkdocs build -d site

.PHONY: all-docs



all-images:
	-mkdir $@


all-images/%.png: | all-images
	cd all-images/ && wget localhost:8000/image/$(basename $(@F)).mov
	mv all-images/$(basename $(@F)).mov $@


ifneq ($(shell nc -z localhost 8000 2>&1),)
dl-all-images: $(addprefix all-images/,$(addsuffix .png,$(basename $(shell curl localhost:8000/images | jq -r '.index | join(" ")'))))

else
dl-all-images:
	$(warning first run a uvicorn server of main.py to be able to download all images)
endif
.PHONY: dll-all-images
