from typing import Dict

from ray.tune.search.optuna import OptunaSearch


class MyOptunaSearch(OptunaSearch):
    def on_trial_result(self, trial_id: str, result: Dict):
        # Check if metric exists in result
        if self._metric not in result:
            return

        # Defensive check: ensure trial_id exists in _ot_trials
        if trial_id not in self._ot_trials:
            # This shouldn't happen now that we fixed trial_str_creator,
            # but keep this check to avoid crashes if it does
            return

        super().on_trial_result(trial_id, result)

    def on_trial_complete(self, trial_id: str, result: Dict = None, error: bool = False):
        # Defensive check for trial completion as well
        if trial_id not in self._ot_trials:
            return

        super().on_trial_complete(trial_id, result, error)
