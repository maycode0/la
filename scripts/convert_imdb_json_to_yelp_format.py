import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a JSON dataset with label/text fields to 'label text' format."
    )
    parser.add_argument("--input", required=True, help="Input JSON path")
    parser.add_argument("--output", required=True, help="Output text path")
    return parser.parse_args()


def normalize_text(text: str) -> str:
    return str(text).replace("\r", " ").replace("\n", " ")


def main() -> None:
    args = parse_args()
    src = Path(args.input)
    dst = Path(args.output)

    data = json.loads(src.read_text(encoding="utf-8"))
    lines = []
    for item in data:
        label = int(item["label"])
        text = normalize_text(item["text"])
        lines.append(f"{label} {text}")

    dst.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(lines)} lines to {dst}")


if __name__ == "__main__":
    main()
