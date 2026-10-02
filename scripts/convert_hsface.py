"""Convert the HSFace face embedder to LiteRT: assets/hsface10k.tflite.

The one model the demo cannot fetch ready-made (emulator/models.py): HSFace
has PyTorch weights on the Hub but no LiteRT build. This makes one, once:

    uv run --with torch --with litert-torch python scripts/convert_hsface.py

The weights are the Vec2Face authors' FR model trained on HSFace10K
(BooBooWu/Vec2Face, fr_weights/hsface10k.pth); the network definition is
their iResNet-50 with squeeze-and-excitation blocks, from
github.com/HaiyuWu/SOTA-Face-Recognition-Train-and-Test (MIT), fetched at a
pinned commit rather than copied here. HSFace10K is a synthetic dataset, but
Vec2Face itself was trained on WebFace4M: check the terms before any use
beyond research.

An iResNet is a plain CNN, so it converts cleanly: a fixed [1, 3, 112, 112]
-> [1, 512] graph. The script checks the LiteRT output against PyTorch on
random inputs and refuses to write a model that disagrees.
"""
from __future__ import annotations

import importlib.util
import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "assets" / "hsface10k.tflite"

WEIGHTS_REPO = "BooBooWu/Vec2Face"
WEIGHTS_FILE = "fr_weights/hsface10k.pth"
IRESNET_URL = ("https://raw.githubusercontent.com/HaiyuWu/"
               "SOTA-Face-Recognition-Train-and-Test/"
               "13931914a1bea16cdc958a86c0c5e15d13789e2b/model/iresnet.py")
# Cosine between the PyTorch and LiteRT embeddings of the same input.
MIN_PARITY = 0.999


def load_iresnet_module():
    with urllib.request.urlopen(IRESNET_URL, timeout=30) as resp:
        source = resp.read()
    path = Path(tempfile.mkdtemp()) / "iresnet.py"
    path.write_bytes(source)
    spec = importlib.util.spec_from_file_location("iresnet", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_model():
    import torch
    from huggingface_hub import hf_hub_download

    net = load_iresnet_module().iresnet("50", num_features=512, mode="se")
    state = torch.load(hf_hub_download(WEIGHTS_REPO, WEIGHTS_FILE),
                       map_location="cpu", weights_only=True)
    state = state.get("state_dict", state)
    net.load_state_dict({k.replace("module.", ""): v for k, v in state.items()},
                        strict=True)
    return net.eval()


def cosine(a, b) -> float:
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(a @ b)


def main() -> int:
    import litert_torch
    import torch
    from ai_edge_litert.interpreter import Interpreter

    net = load_model()
    sample = (torch.randn(1, 3, 112, 112),)
    print("converting to LiteRT...", flush=True)
    with tempfile.TemporaryDirectory() as work:
        candidate = Path(work) / OUT.name
        litert_torch.convert(net, sample).export(str(candidate))

        runner = Interpreter(model_path=str(candidate))
        runner.allocate_tensors()
        inp = runner.get_input_details()[0]
        out = runner.get_output_details()[0]
        worst = 1.0
        for _ in range(3):
            x = torch.randn(1, 3, 112, 112)
            with torch.no_grad():
                expected = net(x)[0].numpy()
            runner.set_tensor(inp["index"], x.numpy().astype(np.float32))
            runner.invoke()
            worst = min(worst, cosine(expected, runner.get_tensor(out["index"])[0]))
        print(f"parity with PyTorch: cosine {worst:.5f}", flush=True)
        if worst < MIN_PARITY:
            print(f"refusing to write a model under {MIN_PARITY}", file=sys.stderr)
            return 1
        OUT.parent.mkdir(parents=True, exist_ok=True)
        candidate.replace(OUT)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
