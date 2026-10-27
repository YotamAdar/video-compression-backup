"""Shared independent-frame roundtrip evaluation; add codecs at the factory."""

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

from .data import image_paths, load_image
from .provenance import environment, git, sha256, source_identity


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def build_codec(config, baseline_root):
    if config["codec"] == "turbo_baseline":
        from .codecs.turbo_baseline import TurboBaseline
        return TurboBaseline(baseline_root, config)
    raise ValueError(f"Unknown codec: {config['codec']}")


def evaluate(config: dict, paths: dict, output: Path):
    """Write one new run directory. Existing runs are never overwritten."""
    import numpy as np
    import torch
    from PIL import Image

    baseline_root = Path(paths["turbo_ddcm_root"]).expanduser().resolve()
    dataset_root = Path(paths["dataset_root"]).expanduser().resolve()
    inputs = image_paths(dataset_root)
    # Avoid writing partial runs if the baseline is missing prerequisites.
    from .codecs.turbo_baseline import preflight
    if config["codec"] == "turbo_baseline":
        preflight(baseline_root, config)
    output.mkdir(parents=True, exist_ok=False)
    (output / "bitstreams").mkdir()
    (output / "reconstructions").mkdir()
    research_root = Path(__file__).resolve().parents[2]
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "initializing",
        "config": config,
        "paths": {**paths, "dataset_root": str(dataset_root), "turbo_ddcm_root": str(baseline_root)},
        "environment": environment(),
        "baseline": source_identity(baseline_root),
        "research": {
            "commit": git(research_root, "rev-parse", "HEAD"),
            "status": git(research_root, "status", "--short"),
            "diff": git(research_root, "diff", "HEAD"),
            "source_sha256": {
                str(p.relative_to(research_root)): sha256(p)
                for p in sorted((research_root / "src").rglob("*.py"))
            },
        },
        "inputs": [{"name": p.name, "sha256": sha256(p)} for p in inputs],
        "timing_scope": "encode includes baseline reconstruction; decode includes VAE; model loading and disk IO excluded",
    }
    table = baseline_root / "turbo_ddcm/combs_files" / f"comb_K{config['K']}_M{config['M']}.pkl"
    metadata["baseline"]["combination_table_sha256"] = sha256(table)
    write_json(output / "metadata.json", metadata)
    try:
        codec = build_codec(config, baseline_root)
        metadata["codec"] = codec.metadata()
        metadata["gpu"] = {
            "name": torch.cuda.get_device_name(config["device"]),
            "torch_cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
        }
        # Native payloads need this shared sidecar to be decoded.
        sidecar = {
            "config": config,
            "codec": metadata["codec"],
            "baseline_source_sha256": metadata["baseline"]["source_sha256"],
            "model_commit": metadata["codec"].get("model_commit"),
        }
        write_json(output / "decoder_config.json", sidecar)
        width, height = metadata["codec"]["width"], metadata["codec"]["height"]
        rows = []
        for path in inputs:
            image, original_size = load_image(path, (width, height), config["device"])
            torch.cuda.synchronize()
            start = time.perf_counter()
            payload = codec.encode(image)
            torch.cuda.synchronize()
            encode_seconds = time.perf_counter() - start
            stream_path = output / "bitstreams" / (path.stem + ".turbo_ddcm")
            stream_path.write_bytes(payload)
            persisted = stream_path.read_bytes()
            torch.cuda.synchronize()
            start = time.perf_counter()
            reconstruction = codec.decode(persisted)
            torch.cuda.synchronize()
            decode_seconds = time.perf_counter() - start
            # Score the same quantized RGB output that is saved on disk.
            rgb = ((reconstruction.detach().float().cpu().squeeze(0).permute(1, 2, 0) / 2 + 0.5).clamp(0, 1).numpy() * 255).round().astype(np.uint8)
            Image.fromarray(rgb).save(output / "reconstructions" / (path.stem + ".png"))
            target = (image.detach().float().cpu().squeeze(0).permute(1, 2, 0).numpy() / 2 + 0.5).clip(0, 1)
            mse = float(np.mean((target.astype(np.float64) - rgb.astype(np.float64) / 255) ** 2))
            row = {
                "name": path.name, "original_size": list(original_size),
                "evaluated_size": [width, height], "payload_bytes": len(payload),
                "payload_bpp": 8 * len(payload) / (width * height),
                "mse_0_to_1": mse, "psnr_db": 10 * math.log10(1 / mse) if mse > 0 else None,
                "encode_seconds": encode_seconds, "decode_seconds": decode_seconds,
            }
            rows.append(row)
            write_json(output / "metrics.json", rows)
        total_pixels = len(rows) * width * height
        total_bytes = sum(row["payload_bytes"] for row in rows)
        sidecar_bytes = (output / "decoder_config.json").stat().st_size
        write_json(output / "summary.json", {
            "frames": len(rows), "payload_bytes": total_bytes,
            "decoder_sidecar_bytes": sidecar_bytes,
            "payload_bpp": total_bytes * 8 / total_pixels,
            "bpp_with_shared_sidecar": (total_bytes + sidecar_bytes) * 8 / total_pixels,
            "mean_mse_0_to_1": sum(r["mse_0_to_1"] for r in rows) / len(rows),
            "encode_seconds": sum(r["encode_seconds"] for r in rows),
            "decode_seconds": sum(r["decode_seconds"] for r in rows),
        })
        metadata["status"] = "complete"
    except Exception as error:
        metadata["status"] = "failed"
        metadata["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        write_json(output / "metadata.json", metadata)
