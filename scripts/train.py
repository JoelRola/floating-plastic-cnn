"""Training entry point; dataset split and UGV schema must be verified first."""

import argparse
from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from floating_plastic.data import load_flopwd, load_ugv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--flopwd", type=Path)
    parser.add_argument("--ugv", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--image-limit", type=int)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    flopwd = args.flopwd or Path(config["paths"]["flopwd"])
    limit = args.image_limit if args.image_limit is not None else config.get("image_limit")
    rows = load_flopwd(flopwd, image_limit=limit)
    if args.ugv:
        load_ugv(args.ugv)
    raise SystemExit(
        f"Validated {len(rows)} FloPWD label rows (seed={args.seed if args.seed is not None else config['seed']}). "
        "Training orchestration is pending a verified split protocol and image/label audit; no model was trained."
    )


if __name__ == "__main__":
    main()
