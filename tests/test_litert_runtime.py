"""Pure helpers of litert_runtime: parsing tensor metadata and reading a buffer.

No real hardware and no model loading — these are pure-function unit tests
that complement live CompiledRunner runs on the board (the detector/ASR are
tested live, not here).
"""

import numpy as np
import pytest
from ai_edge_litert.compiled_model import HardwareAccelerator

from emulator import litert_runtime
from emulator.litert_runtime import (RUNNER_ENV, InterpreterRunner, _meta,
                                     _read, compiled_model_runs_here,
                                     parse_cpu_features)


# --- _meta: normalizing shape/dtype from tensor details ---

def test_meta_normalizes_shape_to_tuple_of_int():
    detail = {"shape": [1, 3, 4], "dtype": np.float32}
    meta = _meta(detail)
    assert meta["shape"] == (1, 3, 4)
    assert all(isinstance(x, int) for x in meta["shape"])


def test_meta_accepts_numpy_int_shape_entries():
    # LiteRT returns shape as a numpy array/numpy int, not a plain int —
    # without an explicit cast, np.prod(shape) can later misbehave with
    # incompatible types, so _meta casts them to int right away.
    detail = {"shape": np.array([1, 64], dtype=np.int64), "dtype": np.int32}
    meta = _meta(detail)
    assert meta["shape"] == (1, 64)
    assert all(isinstance(x, int) for x in meta["shape"])


def test_meta_wraps_dtype_as_numpy_dtype():
    detail = {"shape": [2], "dtype": np.float32}
    meta = _meta(detail)
    assert meta["dtype"] == np.dtype(np.float32)
    assert isinstance(meta["dtype"], np.dtype)


# --- _read: flat buffer -> ndarray of the target shape ---

class FakeBuffer:
    """Stub for a LiteRT buffer: read(count, dtype) -> flat ndarray."""

    def __init__(self, flat: np.ndarray):
        self._flat = flat

    def read(self, count, dtype):
        assert count == self._flat.size, "requested the wrong element count"
        return self._flat.astype(dtype)


def test_read_reshapes_flat_buffer_to_meta_shape():
    meta = {"shape": (2, 3), "dtype": np.dtype(np.float32)}
    flat = np.arange(6, dtype=np.float32)
    out = _read(FakeBuffer(flat), meta)
    assert out.shape == (2, 3)
    assert (out == flat.reshape(2, 3)).all()


def test_read_computes_count_as_product_of_shape():
    # YOLO26n's head: [1, 300, 6] -> 1800 elements in a single flat read.
    meta = {"shape": (1, 300, 6), "dtype": np.dtype(np.float32)}
    flat = np.zeros(1 * 300 * 6, dtype=np.float32)
    out = _read(FakeBuffer(flat), meta)
    assert out.shape == (1, 300, 6)


def test_read_passes_meta_dtype_type_to_buffer_read():
    calls = {}

    class RecordingBuffer:
        def read(self, count, dtype):
            calls["count"] = count
            calls["dtype"] = dtype
            return np.zeros(count, dtype=dtype)

    meta = {"shape": (1, 4), "dtype": np.dtype(np.int32)}
    _read(RecordingBuffer(), meta)
    assert calls["count"] == 4
    assert calls["dtype"] == np.int32


# --- choosing a runner the board can actually run ---

CM4_CPUINFO = """processor\t: 0
BogoMIPS\t: 108.00
Features\t: fp asimd evtstrm crc32 cpuid
CPU implementer\t: 0x41
"""

# A Cortex-A76 (Raspberry Pi 5): the crypto extensions are there.
CRYPTO_CPUINFO = """processor\t: 0
BogoMIPS\t: 108.00
Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32 cpuid
CPU implementer\t: 0x41
"""


def test_parse_cpu_features_reads_the_features_line():
    assert parse_cpu_features(CM4_CPUINFO) == {
        "fp", "asimd", "evtstrm", "crc32", "cpuid"}


def test_parse_cpu_features_without_a_features_line_is_empty():
    # x86 /proc/cpuinfo calls it "flags"; an empty set then means "unknown",
    # and the caller below treats an aarch64 board with no features as unable
    # rather than guessing.
    assert parse_cpu_features("processor\t: 0\nflags\t: fpu vme\n") == set()


def test_a_cpu_without_the_crypto_extensions_cannot_build_a_compiled_model():
    # The robot's CM4. Building one loads libLiteRtWebGpuAccelerator.so, which
    # the aarch64 wheels compile with those extensions; on this CPU it prints
    # "FATAL ERROR: This binary was compiled with aes enabled" and kills the
    # process, so the choice has to be made from the feature list beforehand.
    assert compiled_model_runs_here(
        features=parse_cpu_features(CM4_CPUINFO),
        system="Linux", machine="aarch64", env={}) is False


def test_a_cpu_with_them_can():
    assert compiled_model_runs_here(
        features=parse_cpu_features(CRYPTO_CPUINFO),
        system="Linux", machine="aarch64", env={}) is True


def test_a_machine_that_is_not_linux_aarch64_is_assumed_capable():
    # The Mac this repo is developed on: no /proc/cpuinfo to read, and the
    # wheel it installs is not the aarch64 Linux one.
    assert compiled_model_runs_here(
        features=set(), system="Darwin", machine="arm64", env={}) is True


def test_the_environment_can_force_the_interpreter():
    assert compiled_model_runs_here(
        features=parse_cpu_features(CRYPTO_CPUINFO), system="Linux",
        machine="aarch64", env={RUNNER_ENV: "interpreter"}) is False


def test_the_environment_can_force_the_compiled_model():
    assert compiled_model_runs_here(
        features=parse_cpu_features(CM4_CPUINFO), system="Linux",
        machine="aarch64", env={RUNNER_ENV: "COMPILED"}) is True


# --- InterpreterRunner: the same two calls, over the older API ---

class FakeSignatureRunner:
    def __init__(self, inputs, outputs):
        self._inputs = inputs
        self.outputs = outputs
        self.called_with = None

    def get_input_details(self):
        return self._inputs

    def __call__(self, **inputs):
        self.called_with = inputs
        return self.outputs


class FakeInterpreter:
    """Stub for ai_edge_litert.interpreter.Interpreter."""

    def __init__(self, model_path=None, num_threads=None, signatures=None,
                 inputs=None, outputs=None):
        self.model_path = model_path
        self.num_threads = num_threads
        self._signatures = signatures if signatures is not None else {}
        self._runners = {
            key: FakeSignatureRunner(
                {"in": {"shape": [1, 2], "dtype": np.float32}},
                {"out": np.zeros((1, 2), dtype=np.float32)})
            for key in self._signatures}
        self._in = inputs or []
        self._out = outputs or []
        self.allocated = False
        self.set_tensors = {}
        self.invoked = 0

    def get_signature_list(self):
        return self._signatures

    def get_signature_runner(self, key):
        return self._runners[key]

    def allocate_tensors(self):
        self.allocated = True

    def get_input_details(self):
        return self._in

    def get_output_details(self):
        return self._out

    def set_tensor(self, index, value):
        self.set_tensors[index] = value

    def invoke(self):
        self.invoked += 1

    def get_tensor(self, index):
        return np.full((1, 2), float(index), dtype=np.float32)


def _interpreter_runner(monkeypatch, fake):
    monkeypatch.setattr("ai_edge_litert.interpreter.Interpreter",
                        lambda model_path, num_threads: fake)
    return InterpreterRunner("model.tflite", threads=4)


def test_interpreter_runner_normalizes_signature_input_details(monkeypatch):
    fake = FakeInterpreter(signatures={"encode": {}})
    runner = _interpreter_runner(monkeypatch, fake)
    details = runner.signature("encode").get_input_details()
    # Same normalization as the compiled path: tuple shape, numpy dtype.
    assert details == {"in": {"shape": (1, 2), "dtype": np.dtype(np.float32)}}


def test_interpreter_runner_passes_inputs_through_and_returns_outputs(monkeypatch):
    fake = FakeInterpreter(signatures={"encode": {}})
    runner = _interpreter_runner(monkeypatch, fake)
    sig = runner.signature("encode")
    value = np.ones((1, 2), dtype=np.float32)
    out = sig(**{"in": value})
    assert list(out) == ["out"]
    assert (fake._runners["encode"].called_with["in"] == value).all()


def test_interpreter_runner_only_returns_the_single_signature(monkeypatch):
    fake = FakeInterpreter(signatures={"serving_default": {}})
    runner = _interpreter_runner(monkeypatch, fake)
    assert runner.only().get_input_details() == {
        "in": {"shape": (1, 2), "dtype": np.dtype(np.float32)}}


def test_interpreter_runner_only_drives_tensors_when_nothing_is_named(monkeypatch):
    # The YOLO26n export names its signature, but a model with none is driven
    # by tensor index — the same case _IndexSignature covers for CompiledModel.
    fake = FakeInterpreter(
        signatures={},
        inputs=[{"index": 7, "shape": [1, 3], "dtype": np.float32}],
        outputs=[{"index": 9, "shape": [1, 2], "dtype": np.float32}])
    runner = _interpreter_runner(monkeypatch, fake)
    sig = runner.only()
    assert sig.get_input_details() == {
        "in0": {"shape": (1, 3), "dtype": np.dtype(np.float32)}}
    out = sig(in0=np.ones((1, 3), dtype=np.float32))
    assert fake.allocated and fake.invoked == 1
    assert (fake.set_tensors[7] == np.ones((1, 3), dtype=np.float32)).all()
    assert (out["out0"] == np.full((1, 2), 9.0, dtype=np.float32)).all()


def test_interpreter_runner_is_never_accelerated(monkeypatch):
    fake = FakeInterpreter(signatures={"serving_default": {}})
    assert _interpreter_runner(monkeypatch, fake).is_fully_accelerated() is False


def test_interpreter_runner_refuses_a_model_with_several_signatures(monkeypatch):
    fake = FakeInterpreter(signatures={"encode": {}, "decode": {}})
    runner = _interpreter_runner(monkeypatch, fake)
    with pytest.raises(ValueError, match="one signature"):
        runner.only()


# --- build_runner: the pick, and the refusal ---

def test_build_runner_uses_the_compiled_model_where_the_cpu_allows_it(monkeypatch):
    monkeypatch.setattr(litert_runtime, "compiled_model_runs_here",
                        lambda: True)
    built = {}
    monkeypatch.setattr(litert_runtime, "CompiledRunner",
                        lambda path, accel, threads: built.setdefault(
                            "args", (path, accel, threads)))
    litert_runtime.build_runner("model.tflite", threads=4)
    assert built["args"] == ("model.tflite", HardwareAccelerator.CPU, 4)


def test_build_runner_falls_back_to_the_interpreter_where_it_does_not(monkeypatch):
    monkeypatch.setattr(litert_runtime, "compiled_model_runs_here",
                        lambda: False)
    built = {}
    monkeypatch.setattr(litert_runtime, "InterpreterRunner",
                        lambda path, threads: built.setdefault(
                            "args", (path, threads)))
    litert_runtime.build_runner("model.tflite", threads=4)
    assert built["args"] == ("model.tflite", 4)


def test_build_runner_refuses_the_gpu_on_a_board_that_cannot_load_it(monkeypatch):
    # Without this the process would be killed inside the accelerator with a
    # bare FATAL ERROR and no hint of the cause.
    monkeypatch.setattr(litert_runtime, "compiled_model_runs_here",
                        lambda: False)
    with pytest.raises(SystemExit) as exc:
        litert_runtime.build_runner("model.tflite",
                                    accel=HardwareAccelerator.GPU)
    assert "crypto extensions" in str(exc.value)
    assert RUNNER_ENV in str(exc.value)
