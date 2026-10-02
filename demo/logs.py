"""One logging setup for every entry point: this project's own INFO lines and
everyone's warnings, with a timestamp. Without it, Python drops INFO and
formats the rest bare."""
from __future__ import annotations

import logging

OWN_PACKAGES = ("demo", "emulator")


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    for name in OWN_PACKAGES:
        logging.getLogger(name).setLevel(level)
