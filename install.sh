#!/usr/bin/env bash
# Install GNN Benchmark runtime requirements.
#
# Usage:
#   ./install.sh            install into the active Python environment
#   ./install.sh --venv     create ./venv first and install into it
#   ./install.sh --help     show this help
#
# The script upgrades pip, installs everything in requirements.txt, then
# verifies that each required package imports cleanly. It exits non-zero
# on any failure so it can be used in CI.

set -euo pipefail

create_venv=0
for arg in "$@"; do
    case "$arg" in
        --venv)
            create_venv=1
            ;;
        -h|--help)
            sed -n '2,12p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown option: $arg" >&2
            echo "Run with --help for usage." >&2
            exit 1
            ;;
    esac
done

cd "$(dirname "$0")"

if [ "$create_venv" -eq 1 ]; then
    if [ ! -d venv ]; then
        echo "[install] Creating virtualenv at ./venv ..."
        python3 -m venv venv
    fi
    # shellcheck disable=SC1091
    source venv/bin/activate
fi

echo "[install] Using $(python -V) at $(command -v python)"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "[install] Verifying imports ..."
python - <<'PY'
import importlib
import sys

required = [
    "pandas",
    "numpy",
    "scipy",
    "torch",
    "torchdiffeq",
    "gdown",
    "pyarrow",
    "tables",
]
missing = []
for mod in required:
    try:
        importlib.import_module(mod)
        print(f"  ok   {mod}")
    except ImportError as exc:
        missing.append((mod, str(exc)))
        print(f"  FAIL {mod} ({exc})")

if missing:
    print(f"\n[install] {len(missing)} package(s) failed to import.", file=sys.stderr)
    sys.exit(1)
PY

echo "[install] Done. You can now run:"
echo "  python test_lastvalue_all_datasets.py"
echo "  python test_all_models_beijing_air.py"
