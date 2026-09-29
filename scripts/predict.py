"""Single-image inference entry point."""

import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.parse_args()
    raise SystemExit("Prediction is unavailable until a validated model artifact and preprocessing contract exist.")


if __name__ == "__main__":
    main()
