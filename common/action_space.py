import json
from os import PathLike

from grid2op.Action import BaseAction, ActionSpace
from gymnasium.spaces import Discrete

from .constants import logger


class ReducedActionSpace(Discrete):
    """
    A reduced action space only allowing a subset of all actions.
    """

    def __init__(self, allowed_actions: list[BaseAction]):
        """
        Constructor.

        :param allowed_actions: A list of allowed grid2op actions
        """
        # get all possible single-substation bus change actions
        super().__init__(len(allowed_actions))
        self._allowed_actions = allowed_actions
        logger.info(f"Using reduced action space with {len(allowed_actions)} actions")

    def from_gym(self, action_index: int):
        """
        Transforms an action index from the reduced action space to a grid2op action.

        :param action_index: The action index in the reduced action space
        :return: The corresponding grid2op action
        """
        return self._allowed_actions[action_index]

    def close(self):
        """
        You just have to love grid2op
        """
        pass


class ReducedActionSpace_(ReducedActionSpace):
    def __init__(self, path: PathLike, grid2op_action_space: ActionSpace):
        """
        Constructor.

        :param path: The path to the .json file containing the allowed actions
        """
        allowed_actions = load_actions(path, grid2op_action_space)
        do_nothing_action = grid2op_action_space({})
        allowed_actions.append(do_nothing_action)
        super().__init__(allowed_actions)


def load_actions(path: PathLike, grid2op_action_space: ActionSpace) -> list[BaseAction]:
    """
    Loads the .json with specified topology actions.
    """
    with open(path, "rt", encoding="utf-8") as action_set_file:
        return list(
            (
                grid2op_action_space(action_dict)
                for action_dict in json.load(action_set_file)
            )
        )