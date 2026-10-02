"""The model catalog: every model by name, and where its file comes from.

Swapping a model is a one-line change there; these tests pin that the names
resolve, that the stage defaults point at real entries, and that fetch()
finds a model offline before it ever reaches for the network.
"""

import pytest

from emulator import models


def test_get_returns_the_named_model():
    m = models.get("yolo26n")
    assert m.repo == "Arm/yolo26n-fp16-litert"
    assert m.file == "yolo26n_conv2d_f16_weights.tflite"


def test_get_unknown_name_raises_and_lists_known_names():
    with pytest.raises(KeyError) as exc:
        models.get("does-not-exist")
    assert "yolo26n" in str(exc.value)


def test_every_stage_default_resolves():
    for name in (models.DETECTOR, models.ASR, models.LLM, models.TTS):
        assert isinstance(models.get(name), models.Model)


def test_every_hub_model_names_what_to_download():
    for name, spec in models.MODELS.items():
        if spec.path is None:
            assert spec.repo and (spec.file or spec.patterns), name


def test_the_voice_comes_with_its_runtime_and_frontend():
    patterns = models.get(models.TTS).patterns
    assert "say.py" in patterns
    assert any(p.startswith("frontend") for p in patterns)


def test_fetch_a_local_model_that_is_missing_says_how_to_get_it(tmp_path):
    spec = models.Model(path=tmp_path / "nope.tflite", how_to_get="run the converter")
    with pytest.raises(FileNotFoundError, match="run the converter"):
        models.fetch(spec)


def test_fetch_tries_the_cache_before_the_network(monkeypatch, tmp_path):
    calls = []

    def fake_download(repo_id, filename, local_files_only=False):
        calls.append(local_files_only)
        return str(tmp_path / filename)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    path = models.fetch(models.Model(repo="org/repo", file="m.tflite"))
    assert path == tmp_path / "m.tflite"
    assert calls == [True], "a cached model must load without the network"


def test_fetch_downloads_what_the_cache_does_not_have(monkeypatch, tmp_path):
    calls = []

    def fake_download(repo_id, filename, local_files_only=False):
        calls.append(local_files_only)
        if local_files_only:
            raise FileNotFoundError("not cached")
        return str(tmp_path / filename)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    models.fetch(models.Model(repo="org/repo", file="m.tflite"))
    assert calls == [True, False]


def test_fetch_completes_a_snapshot_that_is_only_partly_cached(monkeypatch, tmp_path):
    # An offline snapshot answers with the folder even when only some of its
    # files were ever downloaded; the named ones must all be there.
    (tmp_path / "say.py").write_text("")
    calls = []

    def fake_snapshot(repo_id, allow_patterns, local_files_only=False):
        calls.append(local_files_only)
        return str(tmp_path)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot)
    models.fetch(models.Model(repo="org/repo",
                              patterns=("say.py", "weights.tflite")))
    assert calls == [True, False]


def test_resolve_llm_with_a_catalog_name_matches_get():
    assert models.resolve_llm(models.LLM) == models.get(models.LLM)


def test_resolve_llm_with_an_unknown_bare_name_raises_key_error():
    with pytest.raises(KeyError):
        models.resolve_llm("not-a-catalog-name-or-a-path")


def test_resolve_llm_with_a_litertlm_path_returns_that_file(tmp_path):
    model_file = tmp_path / "custom.litertlm"
    model_file.write_bytes(b"")
    assert models.resolve_llm(str(model_file)) == models.Model(path=model_file)


def test_resolve_llm_expands_a_home_relative_path(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    model_file = tmp_path / "home.litertlm"
    model_file.write_bytes(b"")
    assert models.resolve_llm("~/home.litertlm").path == model_file


def test_resolve_llm_missing_litertlm_path_raises():
    with pytest.raises(SystemExit):
        models.resolve_llm("/no/such/model.litertlm")


def test_resolve_llm_rejects_a_directory_named_like_a_litertlm_file(tmp_path):
    fake = tmp_path / "looks-like-a-model.litertlm"
    fake.mkdir()
    with pytest.raises(SystemExit):
        models.resolve_llm(str(fake))


def test_a_missing_local_model_is_built_by_its_command(monkeypatch, tmp_path):
    # The face embedder has no LiteRT build to download: fetch makes it.
    target = tmp_path / "built.tflite"
    ran = []

    def build(cmd):
        ran.append(cmd)
        target.write_bytes(b"model")

    monkeypatch.setattr(models, "_build", build)
    spec = models.Model(path=target, build=("make", "it"))
    with pytest.raises(FileNotFoundError):
        models.fetch(spec)                    # not asked to build: fails fast
    assert ran == []
    assert models.fetch(spec, build=True) == target
    assert models.fetch(spec, build=True) == target
    assert ran == [("make", "it")], "built once, then found"


def test_a_build_that_fails_says_how_to_get_the_model(monkeypatch, tmp_path):
    monkeypatch.setattr(models, "_build", lambda cmd: None)
    spec = models.Model(path=tmp_path / "x.tflite", build=("make", "it"),
                        how_to_get="run the converter")
    with pytest.raises(FileNotFoundError, match="run the converter"):
        models.fetch(spec, build=True)


def test_a_build_that_exits_with_an_error_is_a_failure(monkeypatch, tmp_path):
    # A converter that stops half way may leave nothing behind, or may not:
    # its exit code says which, not the file.
    target = tmp_path / "x.tflite"

    def broken(cmd):
        target.write_bytes(b"half")
        return 3

    monkeypatch.setattr(models, "_build", broken)
    spec = models.Model(path=target, build=("make", "it"), how_to_get="run the converter")
    with pytest.raises(FileNotFoundError, match=r"exit 3.*run the converter"):
        models.fetch(spec, build=True)


def test_a_machine_without_the_build_tool_is_told_why(monkeypatch, tmp_path):
    def no_uv(cmd):
        raise FileNotFoundError(2, "No such file or directory", "uv")

    monkeypatch.setattr(models, "_build", no_uv)
    spec = models.Model(path=tmp_path / "x.tflite", build=("uv", "run"),
                        how_to_get="run the converter")
    with pytest.raises(FileNotFoundError, match=r"could not be built.*uv"):
        models.fetch(spec, build=True)


def test_a_named_model_that_cannot_be_had_fails_the_command(monkeypatch, capsys):
    # scripts/robot_service.sh stops the deploy on this exit code.
    def missing(name, build=False):
        raise FileNotFoundError("could not be built")

    monkeypatch.setattr(models, "fetch", missing)
    assert models.main(["hsface"]) == 1
    assert "MISSING" in capsys.readouterr().out


def test_a_missing_model_in_the_repository_is_named_from_its_root():
    spec = models.Model(path=models.ASSETS / "not-there.tflite")
    with pytest.raises(FileNotFoundError) as exc:
        models.fetch(spec)
    assert "assets/not-there.tflite is missing" in str(exc.value)
    assert str(models.REPO) not in str(exc.value)


def test_the_face_embedder_is_built_not_asked_for():
    spec = models.get("hsface")
    assert spec.build and "scripts/convert_hsface.py" in spec.build


def test_named_models_alone_can_be_fetched(monkeypatch, capsys):
    fetched = []
    monkeypatch.setattr(models, "fetch", lambda name, build=False: fetched.append((name, build)) or "path")
    assert models.main(["hsface"]) == 0
    assert fetched == [("hsface", True)]
