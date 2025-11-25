import logging
import tempfile
from pathlib import Path
from typing import Optional

LOGS_PATH = Path("data/logs")
MODELS_PATH = Path("data/models")
NRI_DATASETS_PATH = Path("data/nri_datasets")
EVAL_PATH = Path("data/evaluations")
EDGE_PROBS_PATH = Path("data/edge_probs")

logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def set_experiment_name(experiment_name: Optional[str]):
    global LOGS_PATH, MODELS_PATH, NRI_DATASETS_PATH, EVAL_PATH, EDGE_PROBS_PATH
    if experiment_name is not None:
        LOGS_PATH = Path("data/experiments", experiment_name, "logs")
        MODELS_PATH = Path("data/experiments", experiment_name, "models")
        NRI_DATASETS_PATH = Path("data/experiments", experiment_name, "nri_datasets")
        EVAL_PATH = Path("data/experiments", experiment_name, "evaluations")
        EDGE_PROBS_PATH = Path("data/experiments", experiment_name, "edge_probs")


def enable_test_mode():
    global LOGS_PATH, MODELS_PATH, NRI_DATASETS_PATH, EVAL_PATH, EDGE_PROBS_PATH
    with tempfile.TemporaryDirectory() as tmpdir:
        LOGS_PATH = Path(tmpdir, "logs")
        MODELS_PATH = Path(tmpdir, "models")
        NRI_DATASETS_PATH = Path(tmpdir, "nri_datasets")
        EVAL_PATH = Path(tmpdir, "evaluations")
        EDGE_PROBS_PATH = Path(tmpdir, "edge_probs")