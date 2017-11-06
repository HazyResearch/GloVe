from __future__ import print_function, absolute_import

import logging
import logging.config


def init(level):
    cfg = dict(
              version=1,
              formatters={
                  "f": {"format":
                        "%(levelname)-8s [%(asctime)s] %(message)s",
                        "datefmt":
                        "%m/%d %H:%M:%S"}
                  },
              handlers={
                  "h": {"class": "logging.StreamHandler",
                        "formatter": "f",
                        "level": level}
                  },
              root={
                  "handlers": ["h"],
                  "level": level
                  },
          )

    logging.config.dictConfig(cfg)
