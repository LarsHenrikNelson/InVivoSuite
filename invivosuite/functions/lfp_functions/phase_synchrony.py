import numpy as np


def phase_synchrony(array_1: np.ndarray, array_2: np.ndarray):
    al1 = np.angle(array_1, deg=False)
    al2 = np.angle(array_2, deg=False)
    synchrony = 1 - np.sin(np.abs(al1 - al2) / 2)
    return synchrony
