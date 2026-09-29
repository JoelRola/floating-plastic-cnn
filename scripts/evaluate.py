"""Evaluation entry point reserved for source-aware held-out evaluation."""

import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True, help="Evaluate one dataset/source at a time")
    parser.parse_args()
    raise SystemExit("Evaluation requires a documented held-out split manifest; no mixed-source test set is created.")


if __name__ == "__main__":
    main()
