"""Thin wrapper over the LiteRT CompiledModel API (the current API; Interpreter
is deprecated).

A model is called through one of two signature adapters, both exposing the same
tiny interface — `get_input_details() -> {name: {"shape","dtype"}}` and
`__call__(**inputs) -> {name: ndarray}` — so a stage swaps `Interpreter` for this
with almost no change to its call site, and callers that classify inputs by
shape/dtype (moonshine's `classify_decode_inputs`) keep working unchanged.

Two adapters, because exported `.tflite` files come in two flavours:
- **named signatures** (moonshine: `encode`, `decode`) — driven by `run_by_name`;
- **a single default subgraph, no named signatures** — driven by
  `run_by_index(0, ...)`.

Only fixed-shape models belong here. The Inflect CPU synthesizer is dynamic-shape
(it resizes its input per sentence), which is what CompiledModel is *not* built
for, so it stays on Interpreter — that dynamic shape is exactly why a fixed-chunk
re-export is needed to move it onto CompiledModel.

Buffers are created once and reused; `write()` overwrites them, which matters for
the moonshine decoder that runs its `decode` signature ~16× per utterance.
"""

from __future__ import annotations

import os
import platform

import numpy as np
from ai_edge_litert.compiled_model import CompiledModel, HardwareAccelerator
from ai_edge_litert.options import CpuOptions, Options

# Which runner to build, when the board leaves a choice: "compiled",
# "interpreter", or unset for the capability check below.
RUNNER_ENV = "LITERT_RUNNER"


def parse_cpu_features(text: str) -> set[str]:
    """The Features line of /proc/cpuinfo as a set. Pure, so it is tested."""
    for line in text.splitlines():
        if line.startswith("Features"):
            return set(line.split(":", 1)[1].split())
    return set()


def _read_cpu_features() -> set[str]:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            return parse_cpu_features(fh.read())
    except OSError:
        return set()


def compiled_model_runs_here(*, features: set[str] | None = None,
                             system: str | None = None,
                             machine: str | None = None,
                             env: dict[str, str] | None = None) -> bool:
    """Whether CompiledModel can be built on this machine at all.

    Creating one loads every accelerator the wheel ships, the WebGPU one
    included, before it is asked for the CPU. In the aarch64 wheels that
    library is compiled with the ARMv8 crypto extensions, and on a processor
    without them it does not raise — it prints `FATAL ERROR: This binary was
    compiled with aes enabled` and kills the process. Measured on
    the robot's CM4 (Cortex-A72, `Features: fp asimd evtstrm crc32 cpuid`):
    every CompiledModel died there, the Interpreter ran the same models fine.
    A Cortex-A76 (the Raspberry Pi 5's) has the extensions, and the same
    wheel works there.

    Nothing can be caught after the fact, so the choice is made from the CPU's
    own feature list before anything is created. `LITERT_RUNNER` overrides it
    for a board that disagrees with this rule.
    """
    env = os.environ if env is None else env
    choice = env.get(RUNNER_ENV, "").strip().lower()
    if choice == "interpreter":
        return False
    if choice == "compiled":
        return True
    system = platform.system() if system is None else system
    machine = platform.machine() if machine is None else machine
    if system != "Linux" or machine not in ("aarch64", "arm64"):
        return True
    features = _read_cpu_features() if features is None else features
    return "aes" in features


def _meta(detail) -> dict:
    return {"shape": tuple(int(x) for x in detail["shape"]),
            "dtype": np.dtype(detail["dtype"])}


def _read(buffer, meta) -> np.ndarray:
    count = int(np.prod(meta["shape"]))
    flat = np.asarray(buffer.read(count, meta["dtype"].type))
    return flat.reshape(meta["shape"])


class _NamedSignature:
    """One named signature (moonshine encode/decode), via run_by_name."""

    def __init__(self, model: CompiledModel, key: str) -> None:
        self._model = model
        self._key = key
        sig = model.get_signature_list()[key]
        self._in_names = list(sig["inputs"])
        self._out_names = list(sig["outputs"])
        # get_*_tensor_details(key) returns {signature_name: detail}.
        in_det = model.get_input_tensor_details(key)
        out_det = model.get_output_tensor_details(key)
        self._in = {n: _meta(in_det[n]) for n in self._in_names}
        self._out = {n: _meta(out_det[n]) for n in self._out_names}
        self._in_bufs = {n: model.create_input_buffer_by_name(key, n)
                         for n in self._in_names}
        self._out_bufs = {n: model.create_output_buffer_by_name(key, n)
                          for n in self._out_names}

    def get_input_details(self) -> dict[str, dict]:
        return self._in

    def __call__(self, **inputs) -> dict[str, np.ndarray]:
        for name, array in inputs.items():
            self._in_bufs[name].write(np.ascontiguousarray(array))
        self._model.run_by_name(self._key, self._in_bufs, self._out_bufs)
        return {n: _read(self._out_bufs[n], self._out[n]) for n in self._out_names}


class _IndexSignature:
    """The single default subgraph, no named signature, via run_by_index(0).

    Inputs and outputs are addressed positionally; buffer.get_tensor_details()
    carries the shape and dtype. Names are synthesized (`in0`, `out0`, …) so the
    interface matches _NamedSignature — a single-input model reads back one key.

    The YOLO26n export the detector loads DOES have a named signature (`serving_default`), so `CompiledRunner.only()`
    returns `_NamedSignature` for it (the `len(self._names) == 1` branch), not
    this class. This path exists for the general no-named-signature case and is
    currently exercised only by tests that construct a model with no
    signature list, not by the real detector model.
    """

    def __init__(self, model: CompiledModel, index: int = 0) -> None:
        self._model = model
        self._index = index
        self._in_bufs = model.create_input_buffers(index)
        self._out_bufs = model.create_output_buffers(index)
        self._in_names = [f"in{i}" for i in range(len(self._in_bufs))]
        self._out_names = [f"out{i}" for i in range(len(self._out_bufs))]
        self._in = {n: _meta(b.get_tensor_details())
                    for n, b in zip(self._in_names, self._in_bufs)}
        self._out = {n: _meta(b.get_tensor_details())
                     for n, b in zip(self._out_names, self._out_bufs)}
        self._in_by_name = dict(zip(self._in_names, self._in_bufs))

    def get_input_details(self) -> dict[str, dict]:
        return self._in

    def __call__(self, **inputs) -> dict[str, np.ndarray]:
        for name, array in inputs.items():
            self._in_by_name[name].write(np.ascontiguousarray(array))
        self._model.run_by_index(self._index, self._in_bufs, self._out_bufs)
        return {n: _read(b, self._out[n])
                for n, b in zip(self._out_names, self._out_bufs)}


class CompiledRunner:
    """A CompiledModel plus its signatures.

    `signature(key)` for a named signature (moonshine); `only()` for a
    single-signature or single-default-subgraph model (the YOLO detector),
    which spares the caller from knowing whether the export named its signature.
    """

    def __init__(self, model_path,
                 accel: HardwareAccelerator = HardwareAccelerator.CPU,
                 threads: int = 4) -> None:
        # CompiledModel defaults to CpuOptions(num_threads=1); the old
        # Interpreter path ran 4 threads. Without this the CPU stages regress
        # ~1.75–2.6×. Pass the thread count explicitly.
        if accel == HardwareAccelerator.CPU:
            options = Options(hardware_accelerators=HardwareAccelerator.CPU,
                              cpu_options=CpuOptions(num_threads=threads))
            self._model = CompiledModel.from_file(str(model_path), options=options)
        else:
            self._model = CompiledModel.from_file(str(model_path),
                                                  hardware_accel=accel)
        self._names = list(self._model.get_signature_list())

    def signature(self, key: str) -> _NamedSignature:
        return _NamedSignature(self._model, key)

    def only(self):
        if len(self._names) == 1:
            return _NamedSignature(self._model, self._names[0])
        if not self._names:
            return _IndexSignature(self._model, 0)     # default subgraph, no names
        raise ValueError(
            f"only() needs one signature, model has {self._names}")

    def is_fully_accelerated(self) -> bool:
        return self._model.is_fully_accelerated()


class _InterpreterSignature:
    """One signature of an Interpreter, shaped like _NamedSignature.

    The Interpreter's own signature runner already takes keyword inputs and
    returns a dict of outputs; only the input metadata needs normalizing, so
    that a caller reading `shape`/`dtype` cannot tell the two runners apart.
    """

    def __init__(self, runner) -> None:
        self._runner = runner
        self._in = {name: _meta(detail)
                    for name, detail in runner.get_input_details().items()}

    def get_input_details(self) -> dict[str, dict]:
        return self._in

    def __call__(self, **inputs) -> dict[str, np.ndarray]:
        return self._runner(**inputs)


class _InterpreterTensors:
    """A model with no named signature, driven by tensor indices.

    Names are synthesized (`in0`, `out0`, …) exactly as _IndexSignature does,
    so `only()` returns the same interface whichever runner built it.
    """

    def __init__(self, interpreter) -> None:
        interpreter.allocate_tensors()
        self._it = interpreter
        self._in_details = interpreter.get_input_details()
        self._out_details = interpreter.get_output_details()
        self._in_names = [f"in{i}" for i in range(len(self._in_details))]
        self._out_names = [f"out{i}" for i in range(len(self._out_details))]
        self._in = {n: _meta(d) for n, d in zip(self._in_names, self._in_details)}
        self._by_name = dict(zip(self._in_names, self._in_details))

    def get_input_details(self) -> dict[str, dict]:
        return self._in

    def __call__(self, **inputs) -> dict[str, np.ndarray]:
        for name, array in inputs.items():
            self._it.set_tensor(self._by_name[name]["index"],
                                np.ascontiguousarray(array))
        self._it.invoke()
        return {n: self._it.get_tensor(d["index"])
                for n, d in zip(self._out_names, self._out_details)}


class InterpreterRunner:
    """The same two calls as CompiledRunner, over the older Interpreter API.

    This is the runner for boards where CompiledModel cannot be built at all
    (see compiled_model_runs_here). It is CPU-only by nature: the accelerators
    a CompiledModel would load are exactly what this path avoids.
    """

    def __init__(self, model_path, threads: int = 4) -> None:
        from ai_edge_litert.interpreter import Interpreter  # lazy: CPU path only
        self._it = Interpreter(model_path=str(model_path), num_threads=threads)
        self._names = list(self._it.get_signature_list())

    def signature(self, key: str) -> _InterpreterSignature:
        return _InterpreterSignature(self._it.get_signature_runner(key))

    def only(self):
        if len(self._names) == 1:
            return self.signature(self._names[0])
        if not self._names:
            return _InterpreterTensors(self._it)
        raise ValueError(
            f"only() needs one signature, model has {self._names}")

    def is_fully_accelerated(self) -> bool:
        return False


def build_runner(model_path, *, accel: HardwareAccelerator = HardwareAccelerator.CPU,
                 threads: int = 4):
    """The runner this board can actually run: CompiledRunner where the CPU
    allows it, InterpreterRunner where it does not.

    A GPU request on such a board is refused here, with the reason, rather
    than left to abort the process inside the accelerator it cannot load.
    """
    if compiled_model_runs_here():
        return CompiledRunner(model_path, accel=accel, threads=threads)
    if accel != HardwareAccelerator.CPU:
        raise SystemExit(
            "LiteRT's GPU accelerator cannot run on this processor: it is "
            "built with the ARMv8 crypto extensions and this CPU has none "
            f"(see compiled_model_runs_here, {RUNNER_ENV} overrides). Run the "
            "model on the CPU, or on a board whose CPU has them.")
    return InterpreterRunner(model_path, threads=threads)
