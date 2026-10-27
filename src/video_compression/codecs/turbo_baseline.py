"""Read-only reuse of the local Turbo-DDCM implementation. No numerical patches."""

import ast
import os
from pathlib import Path
import sys

from ..bitstreams.baseline import pack, unpack


def preflight(root: Path, config: dict) -> Path:
    root = root.expanduser().resolve()
    if not (root / "turbo_ddcm/turbo_ddcm.py").is_file():
        raise FileNotFoundError(f"Turbo-DDCM source missing under {root}")
    k, m, t = (config[key] for key in ("K", "M", "T"))
    if not all(type(x) is int for x in (k, m, t)) or not 1 <= m <= k:
        raise ValueError("Require integer 1 <= M <= K and integer T")
    tree = ast.parse((root / "turbo_ddcm/utils.py").read_text())
    schedule = next(
        ast.literal_eval(node.value) for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "SCHEDULER" for target in node.targets)
    )
    if not 2 <= t <= len(schedule):
        raise ValueError(f"Baseline schedule requires 2 <= T <= {len(schedule)}")
    table = root / "turbo_ddcm/combs_files" / f"comb_K{k}_M{m}.pkl"
    if not table.is_file():
        raise FileNotFoundError(
            f"Missing {table}; refusing to let the baseline generate files in its checkout."
        )
    return root


class TurboBaseline:
    def __init__(self, root: Path, config: dict):
        self.root = preflight(root, config)
        self.config = dict(config)
        # Prevent upstream imports from creating __pycache__ in the original tree.
        sys.dont_write_bytecode = True
        # Reuse cached weights only; fail rather than initiating a download.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        existing = sys.modules.get("turbo_ddcm")
        if existing is not None and str(self.root / "turbo_ddcm") not in list(existing.__path__):
            raise RuntimeError("A different turbo_ddcm package is already imported")
        sys.path.insert(0, str(self.root))
        import torch
        from turbo_ddcm.turbo_ddcm import TurboDDCM

        device = config.get("device", "cuda:0")
        if not device.startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError("The unchanged baseline requires an allocated CUDA GPU")
        # Upstream hard-codes self.device='cuda'; align the current CUDA device.
        torch.cuda.set_device(torch.device(device))
        self.device = device
        with torch.inference_mode():
            self.impl = TurboDDCM(
                config["model_id"], config["T"], config["K"], config["M"],
                config["seed"], config.get("float32", False), device,
            )

    def encode(self, image):
        import torch
        with torch.inference_mode():
            (_, bits), _ = self.impl.compress(image, None)
        return pack(bits)

    def decode(self, payload):
        import torch
        with torch.inference_mode():
            return self.impl.decompress(unpack(payload))

    def metadata(self):
        scheduler = self.impl.model.model.scheduler
        return {
            "height": self.impl.H,
            "width": self.impl.W,
            "timesteps": scheduler.timesteps.tolist(),
            "encoding_steps": self.impl.T - 1 - self.impl.no_bits_steps,
            "no_bits_steps": self.impl.no_bits_steps,
            "format": "turbo_ddcm-native-with-external-config",
            "model_commit": getattr(self.impl.model.model.config, "_commit_hash", None),
        }
