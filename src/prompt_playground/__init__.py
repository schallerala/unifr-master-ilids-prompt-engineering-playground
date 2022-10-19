import logging.config
import os
from os import getenv
from pathlib import Path

DIRNAME_SOURCE_PATH = Path(os.path.dirname(__file__)).resolve()

ILIDS_PATH = DIRNAME_SOURCE_PATH.parent.parent.parent / "ilids"

RANDOM_STATE = 16896375


logging_config = {
    "version": 1,
    "loggers": {
        "": {  # root logger
            "level": getenv("LOG_LEVEL", "INFO"),
            "handlers": ["console_handler"],
        },
        "gunicorn": {"propagate": True},
        "gunicorn.error": {
            # "propagate": True,  # /!\ same as uvicorn, this gets overwritten to False in:
            # gunicorn.glogging.Logger.__init__
            # therefore, make sure to declare the expected handler and
            # don't rely on the root logger and its handler(s)
            # Most likely, it is there to overcome the default value of the
            # built-in python library.
            "handlers": ["console_handler"],
        },
        "uvicorn": {"propagate": True},
        "uvicorn.error": {
            # "propagate": True,  # /!\ same as for gunicorn, this gets overwritten to False in:
            # uvicorn.workers.UvicornWorker.__init__
            # therefore, make sure to declare the expected handler and
            # don't rely on the root logger and its handler(s)
            # Most likely, it is there to overcome the default value of the
            # built-in python library.
            "handlers": ["console_handler"],
        },
    },
    "handlers": {
        "console_handler": {
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
    },
}
logging.config.dictConfig(logging_config)

log = logging.getLogger(__name__)
