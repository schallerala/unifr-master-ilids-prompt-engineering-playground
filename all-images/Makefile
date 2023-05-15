%.png:
	wget localhost:8000/image/$(basename $(@F)).mov
	mv $(basename $(@F)).mov $@


ifneq ($(shell nc -z localhost 8000 2>&1),)
dl-all-images: $(addsuffix .png,$(basename $(shell curl localhost:8000/images | jq -r '.index | join(" ")')))

else
dl-all-images:
	$(warning first run a uvicorn server of ../main.py to be able to download all images)
endif
.PHONY: dll-all-images
