"""Build the final submission zip in the structure the challenge requires.

    <team>_submission.zip
    ├── output/matching_results.tsv, output/candidate_pairs.tsv   (from one consistent run)
    ├── code/business_entity_resolution/{src/, colab/, README.md, requirements.txt, run_all.sh}
    └── Documentation_template.md

The two TSVs must come from the SAME run (matches are a subset of the candidates); the
validator is run on them before zipping.

Run:  python src/package.py --run-dir ../../submissions/<folder with both TSVs>
"""
import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

import config

CODE_DIR = Path(__file__).resolve().parents[1]           # code/business_entity_resolution
CODE_FILES = ["README.md", "requirements.txt", "run_all.sh"]
CODE_DIRS = ["src", "colab"]


def build(run_dir, team="Queens_Code", out_dir=None):
    """Validate the run's two TSVs and write <team>_submission.zip."""
    run_dir = Path(run_dir)
    match, cand = run_dir / "matching_results.tsv", run_dir / "candidate_pairs.tsv"
    for f in (match, cand):
        if not f.exists():
            raise FileNotFoundError(f)
    validator = config.DATASET_DIR.parent / "utils" / "validate_submission.py"
    res = subprocess.run([sys.executable, str(validator), "--matching", str(match), "--candidate", str(cand),
                          "--test-dir", str(config.DATASET_DIR / "test"), "--check-ids"],
                         capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(res.stdout.strip()[-400:])
    if res.returncode != 0:
        raise SystemExit("validation failed - not packaging")

    doc = Path(config.WORK_DIR) / "Documentation_template.md"
    out = Path(out_dir or config.WORK_DIR) / f"{team}_submission.zip"
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        z.write(match, "output/matching_results.tsv")
        z.write(cand, "output/candidate_pairs.tsv")
        for name in CODE_FILES:
            z.write(CODE_DIR / name, f"code/business_entity_resolution/{name}")
        for d in CODE_DIRS:
            for f in sorted((CODE_DIR / d).rglob("*")):
                if f.is_file() and "__pycache__" not in f.parts and f.suffix != ".pyc":
                    z.write(f, f"code/business_entity_resolution/{f.relative_to(CODE_DIR).as_posix()}")
        z.write(doc, "Documentation_template.md")
    with zipfile.ZipFile(out) as z:
        names = z.namelist()
    print(f"\nwrote {out} ({out.stat().st_size / 1e6:.0f} MB, {len(names)} files)")
    for n in names:
        print("  ", n)
    return out


def main(argv=None):
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description="Build the submission zip")
    ap.add_argument("--run-dir", required=True, help="folder holding matching_results.tsv + candidate_pairs.tsv")
    ap.add_argument("--team", default="Queens_Code")
    args = ap.parse_args(argv)
    build(args.run_dir, args.team)


if __name__ == "__main__":
    sys.exit(main())
