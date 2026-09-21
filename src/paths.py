from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config" / "project.yaml"
SRC = ROOT / "src"
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
ARTIFACTS = ROOT / "artifacts"
PAPER_TEX = ROOT / "paper"
OUTPUTS = ROOT / "outputs"
SOT = ARTIFACTS / "sot.json"
ORCHESTRATOR_STATUS = OUTPUTS / "logs" / "orchestrator_status.json"
PROTOCOL = ROOT / "config" / "protocol.py"
