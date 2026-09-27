"""Shared file locations for the AVCORP pipeline."""
from pathlib import Path

AVCORP_DIR = Path(__file__).resolve().parent
REPO_ROOT = AVCORP_DIR.parent

MASTER_CSV = AVCORP_DIR / "Deception-Dataset.csv"
TACTICS_KB = AVCORP_DIR / "tactics_knowledge_base.json"

DATASETS_DIR = AVCORP_DIR / "Datasets"
ROLE_HISTORY_DIR = DATASETS_DIR / "role_history"
SEEDS_DIR = DATASETS_DIR / "seeds"
SUMMARIZER_DIR = DATASETS_DIR / "summarizer"
VERIFIED_DIR = DATASETS_DIR / "verified"
REASONING_DIR = DATASETS_DIR / "reasoning"

ASSETS_DIR = REPO_ROOT / "assets"
