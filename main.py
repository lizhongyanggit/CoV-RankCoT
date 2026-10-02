
import os
import sys
import argparse
import subprocess
from pathlib import Path


# ============================================================
# Project configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

CONFIDENCE_SCRIPT = PROJECT_ROOT / "confidence_evaluation.py"
DIAGNOSIS_SCRIPT = PROJECT_ROOT / "query_diagnosis.py"
REWRITE_SCRIPT = PROJECT_ROOT / "query_rewrite.py"


# ============================================================
# Utility
# ============================================================

def check_file(file_path):
    if not file_path.exists():
        raise FileNotFoundError(
            f"Required file not found: {file_path}"
        )


def run_command(command, stage_name):
    print()
    print("=" * 70)
    print(f"Stage: {stage_name}")
    print("=" * 70)

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"{stage_name} failed with return code "
            f"{result.returncode}"
        )

    print(f"{stage_name} completed.")


# ============================================================
# Stage 1
# Multidimensional confidence evaluation
# ============================================================

def run_confidence_evaluation():

    check_file(CONFIDENCE_SCRIPT)

    run_command(
        [
            sys.executable,
            str(CONFIDENCE_SCRIPT)
        ],
        "Multidimensional Confidence Evaluation"
    )


# ============================================================
# Stage 2
# MacBERT-based query defect diagnosis
# ============================================================

def run_query_diagnosis():

    check_file(DIAGNOSIS_SCRIPT)

    run_command(
        [
            sys.executable,
            str(DIAGNOSIS_SCRIPT),
            "--mode",
            "diagnose"
        ],
        "Query Defect Diagnosis"
    )


# ============================================================
# Stage 3
# Category-guided query rewriting and second retrieval
# ============================================================

def run_query_rewrite():

    check_file(REWRITE_SCRIPT)

    run_command(
        [
            sys.executable,
            str(REWRITE_SCRIPT)
        ],
        "Adaptive Query Rewrite and Second Retrieval"
    )


# ============================================================
# Full CoV-RankCoT pipeline
# ============================================================

def run_pipeline():

    print()
    print("CoV-RankCoT Pipeline")
    print()

    print(
        "Step 1: Evaluate the initial answer using "
        "multidimensional confidence."
    )

    run_confidence_evaluation()

    print(
        "Step 2: Diagnose low-confidence queries "
        "using the fine-tuned MacBERT model."
    )

    run_query_diagnosis()

    print(
        "Step 3: Perform category-guided query rewriting "
        "and second retrieval."
    )

    run_query_rewrite()

    print()
    print("=" * 70)
    print("CoV-RankCoT pipeline completed.")
    print("=" * 70)


# ============================================================
# Command-line interface
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "CoV-RankCoT: "
            "Answer-Verification-Driven "
            "Retrieval-Augmented Generation"
        )
    )

    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=[
            "all",
            "confidence",
            "diagnosis",
            "rewrite"
        ],
        help=(
            "Select the pipeline stage to run. "
            "Default: all"
        )
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    if args.stage == "all":

        run_pipeline()

    elif args.stage == "confidence":

        run_confidence_evaluation()

    elif args.stage == "diagnosis":

        run_query_diagnosis()

    elif args.stage == "rewrite":

        run_query_rewrite()


if __name__ == "__main__":
    main()
