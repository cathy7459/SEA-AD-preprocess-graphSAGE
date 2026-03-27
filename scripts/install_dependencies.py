from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS_PATH = PROJECT_ROOT / "requirements.txt"

REQUIRED_IMPORTS = {
    "anndata": "anndata",
    "bs4": "beautifulsoup4",
    "lxml": "lxml",
    "matplotlib": "matplotlib",
    "networkx": "networkx",
    "numpy": "numpy",
    "openpyxl": "openpyxl",
    "pandas": "pandas",
    "psutil": "psutil",
    "pyarrow": "pyarrow",
    "requests": "requests",
    "scanpy": "scanpy",
    "scipy": "scipy",
    "seaborn": "seaborn",
    "sklearn": "scikit-learn",
    "torch": "torch",
    "tqdm": "tqdm",
    "yaml": "pyyaml",
}

SAFE_BOOTSTRAP_PACKAGES = [
    "pip>=24,<26",
    "setuptools>=68,<82",
    "wheel>=0.42,<1",
]


def run_command(command: list[str]) -> None:
    print(f"[install] running: {' '.join(command)}")
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)


def install_requirements(python_executable: str, upgrade_pip: bool) -> None:
    if upgrade_pip:
        run_command([python_executable, "-m", "pip", "install", "--upgrade", *SAFE_BOOTSTRAP_PACKAGES])
    run_command([python_executable, "-m", "pip", "install", "-r", str(REQUIREMENTS_PATH)])


def verify_imports() -> tuple[bool, list[str]]:
    missing: list[str] = []
    for module_name, package_name in REQUIRED_IMPORTS.items():
        try:
            __import__(module_name)
        except Exception as exc:
            missing.append(f"{package_name} (import: {module_name}) -> {exc}")
    return len(missing) == 0, missing


def print_success() -> None:
    print("[install] all required modules imported successfully.")
    print("[install] you can now run the pipeline scripts, for example:")
    print("  python scripts/00_build_cohort_table.py --config configs/project.yaml")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Install all Python dependencies for the SEA-AD GraphSAGE preprocessing pipeline and verify imports."
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use for pip installation. Default: current interpreter.",
    )
    parser.add_argument(
        "--no-upgrade-pip",
        action="store_true",
        help="Skip upgrading pip/setuptools/wheel before installation.",
    )
    args = parser.parse_args()

    if not REQUIREMENTS_PATH.exists():
        raise FileNotFoundError(f"requirements.txt not found at: {REQUIREMENTS_PATH}")

    print(f"[install] project root: {PROJECT_ROOT}")
    print(f"[install] requirements: {REQUIREMENTS_PATH}")
    install_requirements(args.python, upgrade_pip=not args.no_upgrade_pip)

    ok, missing = verify_imports()
    if not ok:
        print("[install] import verification failed for the following packages:")
        for item in missing:
            print(f"  - {item}")
        raise SystemExit(1)

    print_success()


if __name__ == "__main__":
    main()
