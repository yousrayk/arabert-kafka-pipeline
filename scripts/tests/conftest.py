import sys
from pathlib import Path

# scripts/ is a folder of standalone CLIs, not a package: put it on the path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
