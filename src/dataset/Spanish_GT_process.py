import argparse
import re
import os
from pathlib import Path

def process_text(text: str) -> str:
    text = text.replace('-\n', ' ')  # split hyphenated line-ends into two words
    text = text.replace('\n', ' ')   # remove other line breaks
    text = re.sub(r'[^A-Za-zñÑ ]+', '', text)  # remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()   # normalize whitespace
    return text

def process_file(input_path: Path, output_path: Path):
    with input_path.open(encoding="utf-8") as f:
        raw_text = f.read()
    cleaned_text = process_text(raw_text)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write(cleaned_text)
    print(f"Processed {input_path.name} → {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Clean historical Spanish OCR text from file or directory.")
    parser.add_argument("--input_file", type=Path, help="Path to a single input .txt file")
    parser.add_argument("--input_dir", type=Path, help="Path to a directory containing .txt files")
    parser.add_argument("--output_dir", type=Path, help="Directory to save cleaned files (required with --input_dir)")
    args = parser.parse_args()

    if args.input_file and args.input_dir:
        raise ValueError("Use either --input_file or --input_dir, not both.")

    if args.input_file:
        if not args.input_file.exists():
            raise FileNotFoundError(f"Input file {args.input_file} not found.")
        output_path = args.output_dir / args.input_file.name if args.output_dir else args.input_file.with_name(f"{args.input_file.stem}_cleaned.txt")
        process_file(args.input_file, output_path)

    elif args.input_dir:
        if not args.output_dir:
            raise ValueError("You must specify --output_dir when using --input_dir.")
        for txt_file in args.input_dir.glob("*.txt"):
            out_file = args.output_dir / txt_file.name
            process_file(txt_file, out_file)

    else:
        raise ValueError("You must specify either --input_file or --input_dir.")

if __name__ == "__main__":
    main()
