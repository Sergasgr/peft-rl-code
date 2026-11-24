"""Build the fixed KodCode splits once: data/kodcode/{train,valid,test}.jsonl + manifest.json.

    uv run python prepare_data.py

Keeps function tasks whose reference solution passes its own tests, drops prompts that are too long for the
model's tokenizer, and splits with a fixed seed. Takes a while: every reference solution is run.
"""

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from coderl.config import DEFAULTS
from coderl.data import SOURCE, prepare


def main(argv: list[str] | None = None) -> None:
    defaults = json.loads(DEFAULTS.read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=Path("data/kodcode"))
    parser.add_argument("--source", default=SOURCE, help="Hub dataset id, or a local .jsonl/.parquet file")
    parser.add_argument("--revision", default=None, help="dataset revision (default: latest, recorded in the manifest)")
    parser.add_argument("--tokenizer", default=defaults["model_id"])
    parser.add_argument("--n-valid", type=int, default=200)
    parser.add_argument("--n-test", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    manifest = prepare(args.out_dir, args.source, args.revision, args.n_valid, args.n_test, args.seed, tokenizer)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
