import logging
from pathlib import Path

LOGS_PATH = Path("data/logs")
MODELS_PATH = Path("data/models")
NRI_DATASETS_PATH = Path("data/nri_datasets")
EVAL_PATH = Path("data/evaluations")
EDGE_PROBS_PATH = Path("data/edge_probs")

logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)