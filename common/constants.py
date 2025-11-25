import logging
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
    if experiment_name is None:
        LOGS_PATH = Path("data/logs")
        MODELS_PATH = Path("data/models")
        NRI_DATASETS_PATH = Path("data/nri_datasets")
        EVAL_PATH = Path("data/evaluations")
        EDGE_PROBS_PATH = Path("data/edge_probs")
    else:
        LOGS_PATH = Path("data", experiment_name, "logs")
        MODELS_PATH = Path("data", experiment_name, "models")
        NRI_DATASETS_PATH = Path("data", experiment_name, "nri_datasets")
        EVAL_PATH = Path("data", experiment_name, "evaluations")
        EDGE_PROBS_PATH = Path("data", experiment_name, "edge_probs")