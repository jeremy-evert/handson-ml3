from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_PROCESSED = DATA_DIR / "processed"
DATA_ANALYSIS = DATA_DIR / "analysis"

NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"
SCRIPT_DIR = PROJECT_ROOT / "scripts"


def ensure_directories():
    for d in [DATA_RAW, DATA_PROCESSED, DATA_ANALYSIS]:
        d.mkdir(parents=True, exist_ok=True)

