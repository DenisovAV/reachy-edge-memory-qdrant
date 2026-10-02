"""Which way the packages depend: demo/ (the voice loop, its services and
their clients) uses emulator/ (the models and the Qdrant Edge stores), never
the other way round."""
import re
from pathlib import Path

EMULATOR = Path(__file__).resolve().parent.parent / "emulator"


def test_the_models_and_stores_do_not_import_the_demo():
    importing = sorted(path.name for path in EMULATOR.glob("*.py")
                       if re.search(r"^\s*(from|import)\s+demo\b", path.read_text(), re.M))
    assert importing == []
