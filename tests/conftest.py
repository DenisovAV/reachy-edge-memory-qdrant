"""No test builds a real model: emulator/models.py's fetch builds the face
embedder with PyTorch when asked — minutes and gigabytes, which a test that
merely reaches stage.main() or models.main() must never pay. pytest.fail, not
an exception: the callers catch Exception, and the test must still fail."""
import pytest


@pytest.fixture(autouse=True)
def _no_model_builds(monkeypatch):
    from emulator import models

    def refuse(command):
        pytest.fail(f"a test tried to build a model: {command}")

    monkeypatch.setattr(models, "_build", refuse)
