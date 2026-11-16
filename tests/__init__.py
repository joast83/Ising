"""Make the standalone scripts importable during unittest discovery.

Run from the repository root: python3 -m unittest -v
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
