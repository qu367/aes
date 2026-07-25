#!/usr/bin/env python3
"""패치가 이미 반영된 Multi-Scale BERT AES 추론 실행기.

Python 3.8 + transformers==3.4.0 권장 (QWK ≈ 0.797).

데이터: https://github.com/ssuai/asap/tree/main/split 의 p8_test.tsv
모델: Zoho zip → models/p8_3/ (중첩 경로 자동 정리)

Usage:
    python run.py
    python run.py --cpu
    python run.py --allow-transformers4
    python run.py --skip-install-deps
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_URL = (
    "https://files.zohopublic.com.cn/public/workdrive-public/download/"
    "dfpvf0458d50be9664034829928a666b68651?x-cli-msg=null"
)
ASAP_TEST_URL = "https://github.com/ssuai/asap/raw/refs/heads/main/split/p8_test.tsv"
MODEL_MARKER = ROOT / "models" / "p8_3" / "word_document" / "pytorch_model.bin"
DATA_FILE = ROOT / "data" / "p8_test.csv"


def log(msg: str) -> None:
    print(msg, flush=True)


def run_cmd(cmd: list[str], cwd: Path | None = None) -> None:
    log("Running: " + " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(cwd) if cwd else None)


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    log(f"Downloading: {url}")
    log(f"         -> {dest}")
    urllib.request.urlretrieve(url, dest)


def ensure_dependencies(allow_transformers4: bool) -> None:
    if allow_transformers4:
        packages = ["configargparse", "scikit-learn", "scipy", "transformers>=4.30,<5"]
        expected_prefix = None
    else:
        packages = [
            "configargparse",
            "scikit-learn",
            "scipy",
            "transformers==3.4.0",
            "sacremoses",
            "sentencepiece",
        ]
        expected_prefix = "3.4."

    run_cmd([sys.executable, "-m", "pip", "install", "-q", *packages])

    import transformers

    version = transformers.__version__
    log(f"transformers version: {version}")

    if allow_transformers4:
        major = int(version.split(".")[0])
        if major >= 5:
            raise RuntimeError(f"transformers {version} is too new; use 4.x or 3.4.0")
        log("Warning: transformers 4.x mode — QWK will be ~0.53, not 0.797")
        return

    if expected_prefix and not version.startswith(expected_prefix):
        raise RuntimeError(
            f"transformers {version} installed, but transformers==3.4.0 is required "
            "for QWK ~0.797. On Python 3.9+, try: python run.py --allow-transformers4"
        )


def fetch_asap_test_split(dest: Path = DATA_FILE) -> Path:
    """Download ssuai/asap split/p8_test.tsv → id\\ttext\\tscore."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    log(f"Downloading ASAP test split: {ASAP_TEST_URL}")
    with urllib.request.urlopen(ASAP_TEST_URL) as resp:
        raw = resp.read()
    text = raw.decode("latin-1")
    reader = csv.DictReader(io.StringIO(text), delimiter="\t")
    required = ("essay_id", "essay", "domain1_score")
    n = 0
    with open(dest, "w", encoding="utf-8", newline="") as out:
        for row in reader:
            for key in required:
                if key not in row or row[key] is None or row[key] == "":
                    raise ValueError(f"missing column {key!r} in ASAP split row")
            essay = (
                str(row["essay"])
                .replace("\t", " ")
                .replace("\n", " ")
                .replace("\r", " ")
            )
            out.write(f"{row['essay_id']}\t{essay}\t{row['domain1_score']}\n")
            n += 1
    if n == 0:
        raise RuntimeError(f"No rows written from {ASAP_TEST_URL}")
    log(f"Wrote {n} essays → {dest}")
    return dest


def ensure_data() -> Path:
    if DATA_FILE.exists() and DATA_FILE.stat().st_size > 0:
        log(f"Data OK: {DATA_FILE}")
        return DATA_FILE
    return fetch_asap_test_split(DATA_FILE)


def ensure_model() -> None:
    if MODEL_MARKER.exists() and MODEL_MARKER.stat().st_size > 0:
        log(f"Model OK: {MODEL_MARKER}")
        return

    model_dir = ROOT / "models"
    target = model_dir / "p8_3"
    zip_path = ROOT / "Multi-Scale-BERT-AES-Models.zip"
    download_file(MODEL_URL, zip_path)
    log(f"Extracting model zip -> {model_dir}")
    model_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(model_dir)

    # Zoho zip: models/Multi-Scale-BERT-AES-Models/p8_3/...
    nested = model_dir / "Multi-Scale-BERT-AES-Models" / "p8_3"
    if nested.exists() and not MODEL_MARKER.exists():
        if target.exists():
            shutil.rmtree(target)
        nested.rename(target)
        shutil.rmtree(model_dir / "Multi-Scale-BERT-AES-Models", ignore_errors=True)

    # flat: models/word_document + models/chunk
    if not MODEL_MARKER.exists():
        wd = model_dir / "word_document"
        ch = model_dir / "chunk"
        if wd.exists() and ch.exists():
            target.mkdir(parents=True, exist_ok=True)
            for name in ("word_document", "chunk", "config.json", "vocab.txt", "bert_config.json"):
                src = model_dir / name
                if src.exists() and not (target / name).exists():
                    src.rename(target / name)

    if not MODEL_MARKER.exists():
        listing = "\n".join(f"  {p}" for p in sorted(model_dir.rglob("*"))[:40])
        raise FileNotFoundError(
            f"Model not found after download: {MODEL_MARKER}\n"
            f"Contents of {model_dir}:\n{listing}"
        )
    log(f"Model ready: {MODEL_MARKER}")


def write_config(use_cuda: bool) -> Path:
    config_path = ROOT / "asap.ini"
    cuda_line = "cuda\n" if use_cuda else ""
    config_path.write_text(
        f"""prompt: p8
fold: 3
batch_size: 32
data_sample_rate: 1.0
r_dropout: 0
chunk_sizes: 90_30_130_10
{cuda_line}data_dir: {ROOT / 'data'}
test_file: {ROOT / 'data' / 'p8_test.csv'}
model_directory: {ROOT / 'models'}
result_file: {ROOT / 'pred.txt'}
""",
        encoding="utf-8",
    )
    log(f"Wrote config: {config_path}")
    return config_path


def check_environment(use_cuda: bool) -> None:
    import torch

    log(f"Python: {sys.version.split()[0]}")
    log(f"PyTorch: {torch.__version__}")
    log(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name(0)}")
    elif use_cuda:
        log("Warning: --cuda requested but CUDA unavailable; using CPU.")


def print_results(result_file: Path) -> None:
    candidates = [
        result_file,
        ROOT / "models" / "p8_3" / "pred.txt",
        ROOT / "models" / "p8_3" / result_file.name,
    ]
    for cand in candidates:
        if cand.exists():
            result_file = cand
            break
    else:
        raise FileNotFoundError(f"Result file not found: tried {candidates}")

    actuals: list[float] = []
    preds: list[float] = []
    with open(result_file, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 2:
                continue
            actuals.append(float(parts[0]))
            preds.append(float(parts[1]))

    if not actuals:
        raise RuntimeError(f"No predictions found in {result_file}")

    mae = sum(abs(a - p) for a, p in zip(actuals, preds)) / len(actuals)
    log("")
    log("=== Results ===")
    log(f"Samples: {len(actuals)}")
    log(f"Actual score range: {min(actuals):.0f} ~ {max(actuals):.0f}")
    log(f"Predicted score range: {min(preds):.0f} ~ {max(preds):.0f}")
    log(f"MAE: {mae:.2f}")
    log(f"Full results: {result_file}")
    log("Expected QWK on this split: ~0.797 (see qwk: line above)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run patched Multi-Scale BERT AES inference.")
    p.add_argument("--cpu", action="store_true", help="Force CPU")
    p.add_argument(
        "--allow-transformers4",
        action="store_true",
        help="Allow transformers 4.x (QWK ~0.53)",
    )
    p.add_argument(
        "--skip-install-deps",
        action="store_true",
        help="Skip pip install",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    use_cuda = not args.cpu
    result_file = ROOT / "pred.txt"

    log(f"Project directory: {ROOT}")

    if args.allow_transformers4:
        os.environ["AES_ALLOW_TRANSFORMERS4"] = "1"

    if not args.skip_install_deps:
        ensure_dependencies(allow_transformers4=args.allow_transformers4)
    else:
        import transformers

        version = transformers.__version__
        log(f"transformers version: {version}")
        if args.allow_transformers4:
            if int(version.split(".")[0]) >= 5:
                raise RuntimeError(f"transformers {version} not supported")
        elif not version.startswith("3.4."):
            raise RuntimeError(
                f"transformers {version} with --skip-install-deps. "
                "Install transformers==3.4.0 or use --allow-transformers4"
            )

    check_environment(use_cuda)
    ensure_data()
    ensure_model()
    write_config(use_cuda=use_cuda)

    run_cmd([sys.executable, "predict_multi_scale_multi_loss.py"], cwd=ROOT)
    print_results(result_file)


if __name__ == "__main__":
    main()
