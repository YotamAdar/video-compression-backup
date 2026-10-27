import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Compression research (existing environment; no automatic downloads)")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="CPU-only source/config inspection; no model imports")
    run = commands.add_parser("evaluate", help="Run a GPU roundtrip experiment on PNG frames")
    for command in (inspect, run):
        command.add_argument("--config", type=Path, default=Path("configs/baseline.json"))
        command.add_argument("--paths", type=Path, default=Path("configs/paths.local.json"))
    run.add_argument("--name", required=True, help="New run directory name; existing names are rejected")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    paths = json.loads(args.paths.read_text())
    if args.command == "inspect":
        from .codecs.turbo_baseline import preflight
        from .provenance import environment, source_identity
        root = preflight(Path(paths["turbo_ddcm_root"]), config)
        print(json.dumps({"config": config, "baseline": source_identity(root), "environment": environment()}, indent=2))
    else:
        if args.name in (".", "..") or Path(args.name).name != args.name:
            parser.error("--name must be a single directory name")
        from .evaluate import evaluate
        output = Path(paths["runs_root"]).expanduser().resolve() / args.name
        evaluate(config, paths, output)
        print(f"Run saved to {output}")
