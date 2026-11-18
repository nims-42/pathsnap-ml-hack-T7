"""Central settings. Every script imports from here so the team never disagrees on a constant."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
ARTIFACTS = ROOT / "artifacts"
RESULTS.mkdir(exist_ok=True)
ARTIFACTS.mkdir(exist_ok=True)


def _find_data_root():
    """Folder that holds train/ and student_test/. Override with env var PATHSNAP_DATA."""
    env = os.environ.get("PATHSNAP_DATA")
    if env:
        return Path(env)
    for cand in [ROOT / "raw_data",
                 ROOT.parent / "Student folder" / "Student folder",
                 ROOT.parent / "Student folder"]:
        if (cand / "train").exists():
            return cand
    return ROOT / "raw_data"


DATA_ROOT = _find_data_root()
TRAIN_DIR = DATA_ROOT / "train"
STEST_DIR = DATA_ROOT / "student_test"

# projection: the data sits near 55E, 56N, which is UTM zone 40N
CRS_UTM = "EPSG:32640"

SEED = 42
STEP = 5              # keep every 5th GPS point (5 s spacing)
CAND_RADIUS = 50.0    # metres
MAX_CANDS = 8
SAMPLE_SPACING = 10.0 # metres between KD-tree samples along each arc


# route continuity: consecutive predictions count as connected if reachable within this many arcs
CONTINUITY_HOPS = 3
