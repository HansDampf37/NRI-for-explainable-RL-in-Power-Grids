import json
from os import PathLike

from grid2op.Action import BaseAction
from grid2op.Environment import BaseEnv
from gymnasium.spaces import Discrete


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

    def from_gym(self, action_index: int):
        """
        Transforms an action index from the reduced action space to a grid2op action.

        :param action_index: The action index in the reduced action space
        :return: The corresponding grid2op action
        """
        return self._allowed_actions[action_index]


class ReducedActionSpace_(ReducedActionSpace):
    def __init__(self, path: PathLike, env: BaseEnv):
        """
        Constructor.

        :param path: The path to the .json file containing the allowed actions
        """
        allowed_actions = load_actions(path, env)
        do_nothing_action = env.action_space({})
        allowed_actions.append(do_nothing_action)
        super().__init__(allowed_actions)


def load_actions(path: PathLike, env: BaseEnv) -> list[BaseAction]:
    """
    Loads the .json with specified topology actions.
    """
    with open(path, "rt", encoding="utf-8") as action_set_file:
        return list(
            (
                env.action_space(action_dict)
                for action_dict in json.load(action_set_file)
            )
        )