import numpy as np


def random_strategy(choices):
    def strategy(state):
        return np.random.choice(choices)
    return strategy