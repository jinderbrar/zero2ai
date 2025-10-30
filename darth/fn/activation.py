

"""
Generic activation fn that works on any input shape
"""

import numpy as np

def relu(x):
    # No loops, no shape checking needed
    # NumPy handles everything efficiently in C
    return np.maximum(0, x)

def sigmoid(x, clip=(-500, 500)):
    """Numerically stable sigmoid function."""
    x = np.clip(x, clip[0], clip[1])  # Prevent overflow
    return 1 / (1 + np.exp(-x))



